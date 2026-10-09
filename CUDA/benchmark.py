#!/usr/bin/env python3
"""Isolated complete-job 20Ne validation and timing; raw states stay outside Git."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import struct
import subprocess
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def record(handle):
    head = handle.read(4)
    if not head:
        return None
    size = struct.unpack('<i', head)[0]
    if size < 0:
        raise ValueError('Unsupported split Fortran record')
    data = handle.read(size)
    if len(data) != size or handle.read(4) != head:
        raise ValueError('Truncated or invalid Fortran record')
    return data


def checkpoint(path):
    with path.open('rb') as f:
        header = record(f)
        step, timestamp = struct.unpack_from('<id', header)
        states = struct.unpack_from('<i', header, 20)[0]
        grid_record = record(f)
        grid = struct.unpack_from('<iii', grid_record)
        dx, dy, dz, volume = struct.unpack_from('<4d', grid_record, 12)
        neutron_states = struct.unpack_from('<i', header, 40)[0]
        record(f)  # coordinates
        attributes = np.frombuffer(record(f), dtype='<f8').copy().reshape(6, states)
        record(f)  # distribution of states
        psi = np.stack([np.frombuffer(record(f), dtype='<c16').copy() for _ in range(states)])
    if not np.isfinite(psi).all() or not np.isfinite(attributes).all():
        raise ValueError('Nonfinite checkpoint')
    norms = np.sum(np.abs(psi)**2, axis=1) * attributes[0] * volume
    particles = [float(norms[:neutron_states].sum()), float(norms[neutron_states:].sum())]
    return dict(step=step, time=timestamp, states=states, grid=grid, spacing=(dx, dy, dz),
                force=header[12:20].decode().strip(), nucleons=struct.unpack_from('<ii', header, 24),
                psi=psi, attributes=attributes, particles=particles)


def densities(path):
    values = {}
    with path.open('rb') as f:
        record(f)
        record(f)
        while (descriptor := record(f)) is not None:
            name = descriptor[:10].decode().strip()
            data = record(f)
            values[name] = np.frombuffer(data, dtype='<f8').copy()
    return values


def difference(actual, expected):
    delta = actual - expected
    return dict(relative_l2=float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.ravel()), 1e-300)),
                max_abs=float(np.max(np.abs(delta))))


def check_observable(name, actual, reference, initial_flow_atol=5e-6):
    if name == 'energies.res':
        # Total/integrated/kinetic energies retain the tighter tolerance.
        assert np.allclose(actual[:, :3], reference[:, :3], rtol=0, atol=1e-6), name
        assert np.allclose(actual[:, 3:5], reference[:, 3:5], rtol=0, atol=2e-7), name
        assert np.allclose(actual[:, 5], reference[:, 5], rtol=1e-5, atol=1e-7), name
        # j^2/rho is ill-conditioned in padded near-vacuum cells. In this
        # case even CPU fast-math vs strict CPU differs by ~1.4e-6 MeV at t=0.
        initial = np.isclose(actual[:, 0], 0, rtol=0, atol=1e-10)
        assert np.allclose(actual[initial, 6:], reference[initial, 6:], rtol=1e-5, atol=initial_flow_atol), name
        assert np.allclose(actual[~initial, 6:], reference[~initial, 6:], rtol=1e-5, atol=5e-6), name
    else:
        assert np.allclose(actual, reference, rtol=1e-5, atol=1e-7), name


def input_text(steps, grid=(24, 24, 24), validate=False, resetcm=False, amplitude='5.0D-5',
               spacing=(1.0, 1.0, 1.0), mprint=10, dt=0.2, mxpact=4):
    text = (REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/td_k0.in').read_text()
    for key, value in dict(nt=steps, mrest=steps, mprint=mprint, mplot=steps if validate else 0,
                           nof=1, nx=grid[0], ny=grid[1], nz=grid[2], mxpact=mxpact).items():
        text = re.sub(rf'\b{key}\s*=\s*\d+', f'{key}={value}', text, flags=re.I)
    for key, value in zip(('dx', 'dy', 'dz', 'dt'), (*spacing, dt)):
        text, count = re.subn(rf'\b{key}\s*=\s*[\d.+eEdD-]+', f'{key}={value:.17g}', text, flags=re.I)
        if count != 1:
            raise ValueError(f'Expected one {key} in the input template')
    text = text.replace('../../static/20ne_sly5_zaxis.tdhf', '../initial.tdhf')
    text = text.replace("writeselect='r'", "writeselect='rtcsouw', write_isospin=T")
    text = text.replace('ampl_ext=5.0D-5', f'ampl_ext={amplitude}')
    if resetcm:
        text = re.sub(r'\bmxpact=\d+,', f'mxpact={mxpact}, mrescm=1,', text)
    return text


def run(exe, path, text, threads, backend, validate=False, prefix=(), restart=None, particle_atol=1e-6):
    if not np.isfinite(particle_atol) or particle_atol <= 0:
        raise ValueError('Checkpoint particle tolerance must be finite and positive')
    path.mkdir()
    if restart:
        shutil.copy2(restart, path / 'k0_restart.tdhf')
    (path / 'for005').write_text(text)
    env = dict(os.environ, SKY3D_BACKEND=backend, OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1', OMP_PROC_BIND='close', OMP_PLACES='cores')
    env.pop('SKY3D_GPU_VALIDATE', None)
    if validate:
        env['SKY3D_GPU_VALIDATE'] = '1'
    start = time.perf_counter()
    with (path / 'stdout.log').open('w') as log:
        result = subprocess.run([*prefix, str(exe)], cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f'{path}: exit {result.returncode}\n' + (path / 'stdout.log').read_text()[-3000:])
    state = checkpoint(path / 'k0_restart.tdhf')
    requested_steps = int(re.search(r'\bnt=(\d+)', text).group(1))
    if state['step'] != requested_steps:
        raise RuntimeError(f'{path}: incomplete trajectory: saved step {state["step"]}, requested {requested_steps}. '
                           'A legacy Fortran STOP can return exit code zero.\n' + (path/'stdout.log').read_text()[-2000:])
    dt = float(re.search(r'\bdt=([\d.eEdD+-]+)', text).group(1).replace('D', 'E').replace('d', 'e'))
    assert abs(state['time'] - requested_steps*dt) < 1e-8
    assert np.max(np.abs(np.array(state['particles'])-10)) < particle_atol, 'Checkpoint particle number drift'
    energy = np.loadtxt(path / 'energies.res', comments='#', ndmin=2)
    assert np.isfinite(energy).all()
    assert np.max(np.abs(energy[:, 1:3] - 10)) < 1e-5, 'Particle number drift'
    print(path.name, f'{elapsed:.3f}s', flush=True)
    row = dict(directory=str(path), seconds=elapsed, threads=threads, backend=backend,
               input_sha256=hashlib.sha256(text.encode()).hexdigest(),
               final_time=state['time'], final_energy_mev=float(energy[-1, 3]),
               final_checkpoint_particles=state['particles'])
    row['checkpoint_particle_atol'] = particle_atol
    if validate:
        csv = (path / 'gpu_validation.csv').read_text().splitlines()
        row['primitive_validation'] = dict(zip(csv[0].split(','), map(float, csv[1].split(','))))
        assert all(np.isfinite(value) for value in row['primitive_validation'].values())
    return row


def compare(path, reference, initial_flow_atol=5e-6):
    current, expected = checkpoint(path / 'k0_restart.tdhf'), checkpoint(reference / 'k0_restart.tdhf')
    assert current['grid'] == expected['grid'] and current['step'] == expected['step']
    wave = difference(current['psi'], expected['psi'])
    assert wave['relative_l2'] < 1e-9 and wave['max_abs'] < 1e-9, wave
    plots = sorted(path.glob('*.tdd'))
    refs = sorted(reference.glob('*.tdd'))
    actual, target = densities(plots[-1]), densities(refs[-1])
    assert actual.keys() == target.keys()
    fields = {key: difference(actual[key], target[key]) for key in target}
    # Near-zero spin/current fields need absolute tolerances, not unstable relative errors.
    for key, error in fields.items():
        assert error['max_abs'] < 1e-9 + 1e-10 * np.max(np.abs(target[key])), (key, error)
    observables = {}
    for name in ('energies.res', 'quadrupoles.res', 'monopoles.res'):
        a, b = np.loadtxt(path / name, ndmin=2), np.loadtxt(reference / name, ndmin=2)
        observables[name] = difference(a, b)
        if name == 'energies.res':
            observables[name]['total_integrated_energy_max_abs_mev'] = float(np.max(np.abs(a[:, 3:5]-b[:, 3:5])))
        check_observable(name, a, b, initial_flow_atol)
    return dict(wavefunction=wave, fields=fields, printed_observables=observables,
                tolerances=dict(wavefunction_relative_l2=1e-9, wavefunction_max_abs=1e-9,
                                field_max_abs='1e-9 + 1e-10 * max(abs(reference))',
                                printed_observables_rtol=1e-5, printed_observables_atol=1e-7,
                                time_particle_count_rtol=0, time_particle_count_atol=1e-6,
                                total_integrated_energy_rtol=0, total_integrated_energy_atol_mev=2e-7,
                                collective_flow_energy_atol_mev=5e-6,
                                initial_collective_flow_energy_atol_mev=initial_flow_atol,
                                collective_flow_note='j^2/rho in near-vacuum padding is sensitive to roundoff, including between CPU variants'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=REPO / 'CUDA/build')
    parser.add_argument('--state', type=Path)
    parser.add_argument('--prepare-state', action='store_true', help='Generate 20Ne with the included static input')
    parser.add_argument('--static-serr', type=float, default=1e-6,
                        help='Explicit static convergence tolerance for preparation; production input uses 1e-6')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--dt', type=float, default=0.2, help='Explicit timestep; fine meshes may require a smaller value')
    parser.add_argument('--mxpact', type=int, default=4, help='Taylor propagator order, unchanged default 4')
    parser.add_argument('--initial-flow-atol', type=float, default=5e-6,
                        help='Explicit time-zero j^2/rho diagnostic tolerance; later times keep 5e-6 MeV')
    parser.add_argument('--validation-steps', type=int, default=100)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--cpu-threads', type=int, nargs='+', default=[4, 8, 20])
    parser.add_argument('--gpu-threads', type=int, default=8)
    parser.add_argument('--mode', choices=['all', 'validate', 'benchmark', 'sanitizer', 'restart', 'prepare'], default='all')
    parser.add_argument('--output', type=Path, help='Default: results/ for 24^3 or results/20ne-NxNxN/ for another grid')
    args = parser.parse_args()
    if min(args.steps, args.validation_steps, args.repetitions, args.gpu_threads, *args.cpu_threads) < 1:
        parser.error('Step, repetition and thread counts must be positive')
    if not np.isfinite(args.static_serr) or args.static_serr <= 0:
        parser.error('--static-serr must be finite and positive')
    if not np.isfinite(args.dt) or args.dt <= 0 or args.mxpact < 2:
        parser.error('Finite positive --dt and --mxpact >= 2 required')
    if not np.isfinite(args.initial_flow_atol) or args.initial_flow_atol <= 0:
        parser.error('Finite positive --initial-flow-atol required')
    if not args.state and not args.prepare_state:
        parser.error('Use --state PATH or --prepare-state; checkpoints are intentionally excluded from Git')
    root = Path(tempfile.mkdtemp(prefix='sky3d-benchmark-', dir=Path.home() / '.cache'))
    build = args.build_dir.resolve()
    executables = {name: build / name / f'sky3d.{name}' for name in ('reference', 'cpu', 'gpu')}
    if args.prepare_state:
        path = root / 'static'
        path.mkdir()
        text = (REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/static_20ne_sly5.in').read_text()
        text = text.replace('serr=1.0D-6', f'serr={args.static_serr:.16E}'.replace('E', 'D'))
        (path / 'for005').write_text(text)
        with (path / 'stdout.log').open('w') as log:
            subprocess.run([str(executables['cpu'])], cwd=path, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, SKY3D_BACKEND='cpu', OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1'))
        state_path = path / '20ne_sly5.tdhf'
        static_log = (path / 'stdout.log').read_text()
        # Keep convergence evidence; benchmark agreement alone is not physical validation.
        (root / 'static-tail.log').write_text(static_log[-6000:])
        prepared = checkpoint(state_path)
        fluctuation = float(np.dot(prepared['attributes'][0], prepared['attributes'][5]) / prepared['states'])
        if not np.isfinite(fluctuation) or fluctuation >= args.static_serr:
            raise RuntimeError(f'Static state did not meet serr={args.static_serr}: {fluctuation}; inspect {root}/static. '
                               'No tolerance was relaxed automatically; use a validated state or explicitly choose --static-serr for a benchmark.')
    else:
        state_path = args.state.resolve()
    state = checkpoint(state_path)
    if state['states'] != 20 or state['nucleons'] != (10, 10) or state['force'].lower() != 'sly5':
        parser.error('This benchmark requires a Sly5 20Ne / 20-state checkpoint')
    if any(n < 2 or n % 2 for n in state['grid']) or any(not np.isfinite(d) or d <= 0 for d in state['spacing']):
        parser.error('Finite positive spacing and even grid dimensions are required')
    grid = tuple(state['grid'])
    spacing = tuple(state['spacing'])
    def case_input(steps, case_grid=grid, validate=False, resetcm=False, amplitude='5.0D-5'):
        return input_text(steps, case_grid, validate, resetcm, amplitude, spacing=spacing, dt=args.dt, mxpact=args.mxpact)
    if args.output is None:
        args.output = REPO/'CUDA/results'
        if grid != (24, 24, 24):
            args.output /= '20ne-' + 'x'.join(map(str, grid))
    (root / 'initial.tdhf').symlink_to(state_path)
    args.output.mkdir(parents=True, exist_ok=True)
    meta = dict(run_directory=str(root), state_sha256=hashlib.sha256(state_path.read_bytes()).hexdigest(),
                build=json.loads((build / 'build.json').read_text()), nucleus='20Ne', grid=list(grid),
                spacing_fm=list(spacing), states=20,
                dt_fm_c=args.dt, mxpact=args.mxpact,
                initial_flow_atol_mev=args.initial_flow_atol,
                runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                date=time.strftime('%Y-%m-%d'),
                gpu_graphs_enabled=os.environ.get('SKY3D_GPU_GRAPHS', '1') != '0',
                input_template_sha256=hashlib.sha256((REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/td_k0.in').read_bytes()).hexdigest(),
                gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], text=True).strip(),
                cpu=subprocess.check_output(['lscpu'], text=True))
    if args.prepare_state:
        (args.output / 'preparation.json').write_text(json.dumps(dict(**meta, static_iteration=prepared['step'],
             weighted_fluctuation_per_state=fluctuation, serr=args.static_serr, converged_to_requested_tolerance=True,
             state_path=str(state_path), static_run_not_in_tdhf_timings=True), indent=2) + '\n')
        print('Prepared benchmark state meeting requested criterion:', state_path, 'fluctuation:', fluctuation, flush=True)
    if args.mode in ('all', 'validate'):
        cases = [('standard', grid, False, '5.0D-5'),
                 ('rectangular', (grid[0]+4, grid[1]+2, grid[2]), False, '5.0D-5'),
                 ('resetcm', grid, True, '5.0D-5')]
        rows = []
        for label, grid, reset, amplitude in cases:
            text = case_input(args.validation_steps, grid, True, reset, amplitude)
            reference = root / f'{label}-reference'
            run(executables['reference'], reference, text, 1, 'cpu')
            for name, threads in [('cpu', 8), ('gpu', args.gpu_threads)]:
                path = root / f'{label}-{name}'
                timing = run(executables[name], path, text, threads, name, name == 'gpu')
                rows.append(dict(case=label, variant=name, run=timing, comparison=compare(path, reference, args.initial_flow_atol)))
        (args.output / 'validation.json').write_text(json.dumps(dict(**meta, steps=args.validation_steps, cases=rows, passed=True), indent=2) + '\n')
        print('All trajectory validation cases passed', flush=True)
    if args.mode in ('all', 'benchmark'):
        configurations = [('cpu', t) for t in args.cpu_threads] + [('gpu', args.gpu_threads)]
        for name, threads in configurations:
            run(executables[name], root / f'warmup-{name}-{threads}', case_input(20), threads, name)
        rows = []
        for repetition in range(args.repetitions):
            # Rotate/reverse ordering, always execute complete jobs sequentially.
            order = configurations[repetition % len(configurations):] + configurations[:repetition % len(configurations)]
            if repetition % 2:
                order = list(reversed(order))
            for name, threads in order:
                row = run(executables[name], root / f'timing-{repetition}-{name}-{threads}', case_input(args.steps), threads, name)
                rows.append(dict(repetition=repetition, **row))
        medians = {f'{name}-{threads}': statistics.median(row['seconds'] for row in rows
                   if row['backend'] == name and row['threads'] == threads) for name, threads in configurations}
        best_cpu = min(value for name, value in medians.items() if name.startswith('cpu-'))
        speedup = best_cpu / medians[f'gpu-{args.gpu_threads}']
        best_threads = min(args.cpu_threads, key=lambda t: medians[f'cpu-{t}'])
        trajectory_checks = []
        cpu_configuration_checks = []
        for repetition in range(args.repetitions):
            reference = root / f'timing-{repetition}-cpu-{best_threads}'
            for name, threads in configurations:
                path = root / f'timing-{repetition}-{name}-{threads}'
                error = difference(checkpoint(path / 'k0_restart.tdhf')['psi'],
                                   checkpoint(reference / 'k0_restart.tdhf')['psi'])
                assert error['relative_l2'] < 1e-9 and error['max_abs'] < 1e-9, (name, threads, error)
                for file in ('energies.res', 'quadrupoles.res', 'monopoles.res'):
                    check_observable(file, np.loadtxt(path / file, ndmin=2), np.loadtxt(reference / file, ndmin=2), args.initial_flow_atol)
                checks = trajectory_checks if name == 'gpu' else cpu_configuration_checks
                checks.append(dict(repetition=repetition, cpu_threads=best_threads, compared_threads=threads,
                                   wavefunction=error, printed_observables_passed=True))
        (args.output / 'benchmark.json').write_text(json.dumps(dict(**meta, steps=args.steps,
                repetitions=args.repetitions, mprint=10, mplot=0, mrest=args.steps,
                complete_job_startup_and_final_checkpoint_included=True, runs=rows,
                median_seconds=medians, speedup_against_best_measured_cpu=speedup,
                final_trajectory_checks=trajectory_checks,
                cpu_configuration_trajectory_checks=cpu_configuration_checks,
                printed_comparison_tolerances=dict(total_integrated_energy_atol_mev=2e-7,
                    total_integrated_energy_rtol=0, collective_flow_energy_atol_mev=5e-6,
                    initial_collective_flow_energy_atol_mev=args.initial_flow_atol,
                    other_printed_rtol=1e-5, other_printed_atol=1e-7)), indent=2) + '\n')
        print('Medians:', medians, 'GPU speedup against best CPU:', speedup, flush=True)
    if args.mode == 'sanitizer':
        sanitizer = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin/compute-sanitizer'
        path = root / 'sanitizer'
        row = run(executables['gpu'], path, case_input(2, validate=True), 8, 'gpu', True,
                  [str(sanitizer), '--tool', 'memcheck', '--error-exitcode', '1'])
        log = (path / 'stdout.log').read_text()
        assert 'ERROR SUMMARY: 0 errors' in log
        (args.output / 'sanitizer.json').write_text(json.dumps(dict(**meta, run=row, passed=True,
                  summary='Integrated two-step trajectory and initial Hamiltonian/densities: 0 memcheck errors'), indent=2) + '\n')
    if args.mode == 'restart':
        seed = root / 'seed'
        run(executables['reference'], seed, case_input(20), 1, 'cpu')
        text = case_input(40, validate=True).replace('imode=2,', 'imode=2, trestart=T,')
        reference = root / 'restart-reference'
        run(executables['reference'], reference, text, 1, 'cpu', restart=seed / 'k0_restart.tdhf')
        rows = []
        for name, threads in [('cpu', 8), ('gpu', args.gpu_threads)]:
            path = root / f'restart-{name}'
            timing = run(executables[name], path, text, threads, name, name == 'gpu', restart=seed / 'k0_restart.tdhf')
            rows.append(dict(variant=name, run=timing, comparison=compare(path, reference, args.initial_flow_atol)))
        (args.output / 'restart.json').write_text(json.dumps(dict(**meta, cases=rows, passed=True,
                 seed_step=20, final_step=40, note='All variants resumed the same strict CPU checkpoint'), indent=2) + '\n')
    print('Raw output retained at', root, flush=True)


if __name__ == '__main__':
    main()
