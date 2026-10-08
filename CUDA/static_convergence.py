#!/usr/bin/env python3
"""Probe a static checkpoint or run a fixed-box 20Ne mesh study in Linux/WSL.

Instrumentation is compiled in a fresh cache directory, never into Code/.
The production Hamiltonian and stopping criterion are not modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

from benchmark import REPO, checkpoint, record

FLAGS = '-O3 -march=native -ffp-contract=off -fopenmp'
TEMPLATE = REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/static_20ne_sly5.in'
PROBE = '''
  SUBROUTINE convergence_probe
    USE Trivial, ONLY: overlap, rpsnorm
    INTEGER :: nst, unit
    REAL(db) :: energy
    COMPLEX(db), ALLOCATABLE :: hp(:,:,:,:), res(:,:,:,:), hr(:,:,:,:)
    ALLOCATE(hp(nx,ny,nz,2),res(nx,ny,nz,2),hr(nx,ny,nz,2))
    OPEN(NEWUNIT=unit,FILE='probe.bin',ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE')
    WRITE(unit) nx,ny,nz,nstmax,wxyz
    WRITE(unit) wocc,bmass,psi
    DO nst=1,nstmax
       CALL hpsi(isospin(nst),0.D0,psi(:,:,:,:,nst),hp)
       energy=REAL(overlap(psi(:,:,:,:,nst),hp))/rpsnorm(psi(:,:,:,:,nst))
       res=hp-energy*psi(:,:,:,:,nst)
       CALL hpsi(isospin(nst),energy,res,hr)
       WRITE(unit) hp,res,hr
    ENDDO
    CALL sp_properties
    CALL moments(0,0)
    CALL integ_energy
    CALL sum_energy
    WRITE(unit) ehfint,ehf,rmstot,beta,gamma,pnr,cmtot
    CLOSE(unit)
  END SUBROUTINE convergence_probe
'''


def assign(text, name, value):
    text, count = re.subn(r'\b' + name + r'\s*=\s*[^,\n/]+', f'{name}={value}', text, flags=re.I)
    if count != 1:
        raise ValueError(f'Expected exactly one {name} assignment in the template')
    return text


def grid_input(text, grid, spacing):
    for name, value in zip(('nx', 'ny', 'nz'), grid):
        text = assign(text, name, int(value))
    for name, value in zip(('dx', 'dy', 'dz'), spacing):
        text = assign(text, name, f'{value:.16g}')
    return text


def state_geometry(path):
    with path.open('rb') as handle:
        header = record(handle)
        if header[12:20].decode().strip().lower() != 'sly5' or tuple(np.frombuffer(header[20:32], '<i4')) != (20, 10, 10):
            raise ValueError('This diagnostic template requires a Sly5 20Ne / 20-state checkpoint')
        neutrons = int(np.frombuffer(header[40:44], '<i4')[0])
        grid = tuple(np.frombuffer(record(handle)[:12], '<i4'))
        coords = np.frombuffer(record(handle), '<f8')
    if any(n % 2 for n in grid):
        raise ValueError('The spectral probe requires even grid dimensions')
    axes = np.split(coords, np.cumsum(grid)[:-1])
    spacing = tuple(float(axis[1] - axis[0]) for axis in axes)
    if any(not np.allclose(np.diff(axis), step, rtol=0, atol=1e-12)
           for axis, step in zip(axes, spacing)):
        raise ValueError('Nonuniform checkpoint grid')
    return grid, spacing, neutrons


def run(executable, directory, threads):
    start = time.perf_counter()
    with (directory / 'stdout.log').open('w') as log:
        subprocess.run([str(executable)], cwd=directory, check=True, stdout=log,
                       stderr=subprocess.STDOUT, env=dict(os.environ, SKY3D_BACKEND='cpu',
                       OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1'))
    return time.perf_counter() - start


def build_probe(root):
    build = root / 'probe-build'
    build.mkdir()
    sources = {}
    for path in (REPO / 'Code').iterdir():
        if path.suffix in {'.f90', '.f', '.data'} or path.name == 'Makefile':
            shutil.copy2(path, build / path.name)
            sources[str(path.relative_to(REPO))] = hashlib.sha256(path.read_bytes()).hexdigest()
    command = ['make', '-j2', 'makeit', 'COMPILER_SKY=gfortran',
               'PARALLEL_SKY=sequential.o', f'COMPILERFLAGS_SKY={FLAGS}',
               'LIBS_SKY=-lfftw3 -llapack -lopenblas']
    with (build / 'build.log').open('w') as log:
        subprocess.run(command + ['EXEC_SKY=sky3d.static'], cwd=build, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    src = (build / 'static.f90').read_text()
    anchor = '    CALL skyrme\n    ! Step 3:'
    if src.count(anchor) != 1:
        raise ValueError('Static initialization instrumentation anchor changed')
    src = src.replace(anchor, '    CALL skyrme\n    CALL convergence_probe\n    STOP\n    ! Step 3:')
    src, count = re.subn(r'(?im)^END MODULE Static\s*$', PROBE + '\nEND MODULE Static', src)
    if count != 1:
        raise ValueError('Static module instrumentation anchor changed')
    (build / 'static.f90').write_text(src)
    with (build / 'build.log').open('a') as log:
        subprocess.run(command + ['EXEC_SKY=sky3d.probe'],
                       cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
    exe = build / 'sky3d.probe'
    return exe, dict(flags=FLAGS, source_sha256=sources,
                     executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
                     static_executable_sha256=hashlib.sha256((build/'sky3d.static').read_bytes()).hexdigest(),
                     diagnostic_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     compiler=subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0],
                     source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip())


def analyse(path, spacing, neutrons):
    with path.open('rb') as handle:
        nx, ny, nz, states = map(int, np.fromfile(handle, '<i4', 4))
        volume = float(np.fromfile(handle, '<f8', 1)[0])
        occ = np.fromfile(handle, '<f8', states)
        mass = np.fromfile(handle, '<f8', nx*ny*nz*2).reshape((nx, ny, nz, 2), order='F')
        flat = np.fromfile(handle, '<c16', nx*ny*nz*2*states)
        vectors = np.fromfile(handle, '<c16', states*3*nx*ny*nz*2).reshape(states, 3, -1)
        observables = np.fromfile(handle, '<f8', 10)
        if len(observables) != 10 or handle.read(1):
            raise ValueError('Probe format mismatch')
    psi = flat.reshape((nx, ny, nz, 2, states), order='F')
    basis = flat.reshape(states, -1)
    hp, residual, hres = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    if not np.isfinite(vectors).all() or not np.isfinite(flat).all():
        raise ValueError('Nonfinite probe')
    # Independently reconstruct just the expanded effective-mass operator.
    # First derivatives zero the even-grid Nyquist mode, matching Levels;
    # second derivatives retain it. Real-space field derivatives match sder.
    iq = np.array([0]*neutrons + [1]*(states-neutrons))
    mass_hp = np.zeros_like(psi)
    matrix_derivative_error = []
    for axis, (n, step) in enumerate(zip((nx, ny, nz), spacing)):
        k = 2*np.pi*np.fft.fftfreq(n, d=step)
        k1 = k.copy()
        k1[n//2] = 0
        shape = [1]*5
        shape[axis] = n
        fft = np.fft.fft(psi, axis=axis)
        d1 = np.fft.ifft(fft*(1j*k1.reshape(shape)), axis=axis)
        d2 = np.fft.ifft(fft*(-k**2).reshape(shape), axis=axis)
        shape = [1]*4
        shape[axis] = n
        db = np.fft.ifft(np.fft.fft(mass, axis=axis)*(1j*k1.reshape(shape)), axis=axis).real
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
        afac = 2*np.pi/n
        der = sum(-m*np.sin(m*afac*(j-i)) for m in range(1, n//2))
        der -= .5*(n//2)*np.sin((n//2)*afac*(j-i))
        der *= -afac/((n//2)*step)
        matrix_db = np.moveaxis(np.tensordot(der, mass, axes=(1, axis)), 0, axis)
        matrix_derivative_error.append(float(np.max(np.abs(matrix_db-db))))
        mass_hp -= db[:, :, :, iq][:, :, :, None, :]*d1 + mass[:, :, :, iq][:, :, :, None, :]*d2
    mass_flat = mass_hp.reshape((-1, states), order='F').T
    rows = []
    all_residuals = []
    all_h2 = []
    for label, selection in [('neutrons', slice(0, neutrons)), ('protons', slice(neutrons, states))]:
        a, h, r = basis[selection], hp[selection], residual[selection]
        gram = a.conj() @ a.T * volume
        projected = a.conj() @ h.T * volume
        projected_mass = a.conj() @ mass_flat[selection].T * volume
        other = projected - projected_mass
        occupied_residual = a.conj() @ r.T * volume
        outside = r - occupied_residual.T @ a
        direct = np.sqrt(np.sum(np.abs(r)**2, axis=1)*volume)
        h2 = np.sqrt(np.abs(np.sum(a.conj()*hres[selection], axis=1).real*volume))
        all_residuals.extend(direct.tolist())
        all_h2.extend(h2.tolist())
        rows.append(dict(species=label, orthonormality_max_abs=float(np.max(np.abs(gram-np.eye(len(a))))),
                         hamiltonian_antihermiticity_max_abs_mev=float(np.max(np.abs(projected-projected.conj().T))),
                         effective_mass_antihermiticity_max_abs_mev=float(np.max(np.abs(projected_mass-projected_mass.conj().T))),
                         remaining_terms_antihermiticity_max_abs_mev=float(np.max(np.abs(other-other.conj().T))),
                         direct_residual_norm_mev=direct.tolist(), h_squared_fluctuation_mev=h2.tolist(),
                         unoccupied_residual_norm_mev=np.sqrt(np.sum(np.abs(outside)**2, axis=1)*volume).tolist(),
                         orbital_energy_mev=projected.diagonal().real.tolist()))
    return dict(grid=[nx, ny, nz], spacing_fm=list(spacing), states=states,
                fresh_field_observables=dict(integrated_energy_mev=float(observables[0]),
                    koopman_energy_using_saved_orbital_energies_mev=float(observables[1]),
                    rms_radius_fm=float(observables[2]), beta=float(observables[3]), gamma_degrees=float(observables[4]),
                    particles=observables[5:7].tolist(), center_of_mass_fm=observables[7:10].tolist()),
                fft_matrix_field_derivative_max_abs=matrix_derivative_error,
                weighted_direct_residual_per_state_mev=float(np.dot(occ, all_residuals)/states),
                weighted_h_squared_fluctuation_per_state_mev=float(np.dot(occ, all_h2)/states), species=rows,
                note='Fresh fields rebuilt from checkpoint psi; projected occupied-subspace measurements. '
                     'This is not a full operator-norm bound or a physical ground-state validation.')


def probe(root, executable, state, threads):
    case = root / 'probe'
    case.mkdir()
    grid, spacing, neutrons = state_geometry(state)
    if not np.allclose(checkpoint(state)['attributes'][0], 1, rtol=0, atol=1e-12):
        raise ValueError('The probe template requires fully occupied no-pairing orbitals')
    text = grid_input(TEMPLATE.read_text(), grid, spacing)
    text = text.replace('imode=1,', 'imode=1, trestart=T,')
    (case / 'for005').write_text(text)
    (case / '20ne_sly5.tdhf').symlink_to(state.resolve())
    run(executable, case, threads)
    result = analyse(case / 'probe.bin', spacing, neutrons)
    saved = checkpoint(state)
    result.update(checkpoint_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),
                  checkpoint_path=str(state), saved_iteration=saved['step'],
                  checkpoint_saved_h_squared_fluctuation_per_state_mev=float(np.dot(saved['attributes'][0], saved['attributes'][5])/saved['states']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--state', type=Path, help='Probe an existing Sly5 / no-pairing 20Ne static checkpoint')
    group.add_argument('--mesh', type=int, help='Run a fresh cubic 20Ne grid with n*dx=24 fm')
    parser.add_argument('--maxiter', type=int, default=3000)
    parser.add_argument('--settle-iterations', type=int, default=0,
                        help='Optionally continue a COPY of the prepared/provided state before probing')
    parser.add_argument('--settle-serr', type=float, default=1e-8,
                        help='Explicit stricter criterion for optional settling (default 1e-8)')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--executable', type=Path, help='Optional existing static executable; otherwise build strict CPU from current source')
    parser.add_argument('--output', type=Path, required=True, help='Small JSON evidence path')
    args = parser.parse_args()
    if os.name != 'posix':
        parser.error('Run in Linux/WSL')
    if args.threads < 1 or args.maxiter < 1 or (args.mesh and (args.mesh < 8 or args.mesh % 2)):
        parser.error('Positive threads/iterations and an even mesh >= 8 are required')
    if args.settle_iterations < 0 or not np.isfinite(args.settle_serr) or args.settle_serr <= 0:
        parser.error('Nonnegative settle iterations and finite positive settle serr required')
    root = Path(tempfile.mkdtemp(prefix='sky3d-convergence-', dir=Path.home()/'.cache'))
    print('Raw evidence directory:', root, flush=True)
    exe, build = build_probe(root)
    if args.mesh:
        case = root / 'static'
        case.mkdir()
        text = grid_input(TEMPLATE.read_text(), (args.mesh,)*3, (24/args.mesh,)*3)
        for name, value in [('maxiter', args.maxiter), ('mprint', 20), ('mplot', 0), ('mrest', args.maxiter)]:
            text = assign(text, name, value)
        (case/'for005').write_text(text)
        static_exe = args.executable.resolve() if args.executable else exe.parent/'sky3d.static'
        wall = run(static_exe, case, args.threads)
        state = case/'20ne_sly5.tdhf'
        saved = checkpoint(state)
        fluct = float(np.dot(saved['attributes'][0], saved['attributes'][5])/saved['states'])
        preparation = dict(wall_seconds=wall, requested_serr=1e-6, maxiter=args.maxiter,
                           saved_iteration=saved['step'], saved_h_squared_fluctuation_mev=fluct,
                           criterion_met=fluct < 1e-6, executable_sha256=hashlib.sha256(static_exe.read_bytes()).hexdigest(),
                           input=text)
    else:
        state = args.state.resolve()
        preparation = None
    settling = None
    if args.settle_iterations:
        # A restart writes wffile: copy it, never symlink a writable checkpoint.
        case = root/'settled'
        case.mkdir()
        copied = case/'20ne_sly5.tdhf'
        original_hash = hashlib.sha256(state.read_bytes()).hexdigest()
        shutil.copy2(state, copied)
        saved = checkpoint(copied)
        limit = saved['step'] + args.settle_iterations
        grid, spacing, _ = state_geometry(copied)
        text = grid_input(TEMPLATE.read_text(), grid, spacing).replace('imode=1,', 'imode=1, trestart=T,')
        for name, value in [('maxiter', limit), ('mprint', 20), ('mplot', 0), ('mrest', limit),
                            ('serr', f'{args.settle_serr:.16E}'.replace('E', 'D'))]:
            text = assign(text, name, value)
        (case/'for005').write_text(text)
        static_exe = args.executable.resolve() if args.executable else exe.parent/'sky3d.static'
        wall = run(static_exe, case, args.threads)
        if hashlib.sha256(state.read_bytes()).hexdigest() != original_hash:
            raise RuntimeError('Settling modified the supplied checkpoint')
        final = checkpoint(copied)
        fluct = float(np.dot(final['attributes'][0], final['attributes'][5])/final['states'])
        settling = dict(input=text, starting_checkpoint_sha256=original_hash, source_checkpoint_preserved=True,
                        starting_iteration=saved['step'], iteration_limit=limit, wall_seconds=wall,
                        requested_serr=args.settle_serr, saved_iteration=final['step'], criterion_met=fluct < args.settle_serr,
                        executable_sha256=hashlib.sha256(static_exe.read_bytes()).hexdigest())
        state = copied
    result = dict(run_directory=str(root), threads=args.threads, build=build,
                  preparation=preparation, settling=settling, probe=probe(root, exe, state, args.threads))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Saved:', args.output, flush=True)
    print('Fresh-field weighted h**2 fluctuation:', result['probe']['weighted_h_squared_fluctuation_per_state_mev'], flush=True)
    print('Fresh-field weighted direct residual:', result['probe']['weighted_direct_residual_per_state_mev'], flush=True)


if __name__ == '__main__':
    main()
