#!/usr/bin/env python3
"""Extend the settled 20Ne controls; retain raw numerical output in WSL cache."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from benchmark import REPO, checkpoint, compare, difference, input_text, run


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def exact_steps(duration, dt):
    if not np.isfinite(duration) or not np.isfinite(dt) or min(duration, dt) <= 0:
        raise ValueError('Duration and timestep must be finite and positive')
    steps = round(duration / dt)
    if steps < 1 or abs(steps * dt - duration) > 1e-9:
        raise ValueError('Duration must be an integer number of timesteps')
    return steps


def gram(state):
    # ψ includes both spin components; compare each species independently.
    volume = float(np.prod(state['spacing']))
    split = state['nucleons'][0]  # this runner requires ten states per species
    return [volume * p.conj() @ p.T for p in
            (state['psi'][:split], state['psi'][split:])]


def invariants(path, initial_gram, dt, order):
    state = checkpoint(path / 'k0_restart.tdhf')
    matrices = gram(state)
    energy = np.loadtxt(path / 'energies.res', ndmin=2)
    result = dict(
        gram_max_abs_error=float(max(np.max(np.abs(g - np.eye(len(g)))) for g in matrices)),
        gram_max_abs_drift=float(max(np.max(np.abs(g - g0)) for g, g0 in zip(matrices, initial_gram))),
        integrated_energy_initial_mev=float(energy[0, 4]),
        integrated_energy_final_mev=float(energy[-1, 4]),
        integrated_energy_max_drift_mev=float(np.max(np.abs(energy[:, 4] - energy[0, 4]))),
        final_particles=state['particles'], dt_fm_c=dt, taylor_order=order)
    if not all(np.isfinite(g).all() for g in matrices):
        raise AssertionError('Nonfinite overlap matrix')
    if result['gram_max_abs_error'] > 1e-6 or result['gram_max_abs_drift'] > 1e-6:
        raise AssertionError(f'Occupied-state orthogonality/norm drift: {result}')
    # Record physical energy drift; CPU/GPU equivalence is tested separately.
    # There is no automatic acceptance threshold for physical convergence here.
    return result


def integration_difference(actual, reference):
    a, b = checkpoint(actual / 'k0_restart.tdhf'), checkpoint(reference / 'k0_restart.tdhf')
    if a['grid'] != b['grid'] or abs(a['time'] - b['time']) > 1e-8:
        raise ValueError('Integration comparison requires the same grid and physical endpoint')
    result = dict(wavefunction=difference(a['psi'], b['psi']), printed_observables={})
    for name in ('energies.res', 'quadrupoles.res', 'monopoles.res'):
        x, y = np.loadtxt(actual / name, ndmin=2), np.loadtxt(reference / name, ndmin=2)
        if x.shape != y.shape or not np.allclose(x[:, 0], y[:, 0], rtol=0, atol=1e-8):
            raise ValueError(f'Integration comparison requires matched physical output times: {name}')
        result['printed_observables'][name] = difference(x[:, 1:], y[:, 1:])
        if name == 'energies.res':
            result['integrated_energy_max_difference_mev'] = float(np.max(np.abs(x[:, 4] - y[:, 4])))
    result['interpretation'] = 'Measured integration sensitivity, not a CPU/GPU equivalence or production-accuracy pass'
    return result


def verify_build(build):
    metadata = json.loads((build / 'build.json').read_text())
    for name, expected in metadata['source_sha256'].items():
        if sha256(REPO / name) != expected:
            raise ValueError(f'Stale build source: {name}; rebuild before validation')
    executables = {name: build / name / f'sky3d.{name}' for name in ('reference', 'cpu', 'gpu')}
    for name, path in executables.items():
        if sha256(path) != metadata['executable_sha256'][name]:
            raise ValueError(f'Executable hash mismatch: {path}')
    library = build / 'libsky3d_gpu.so'
    if sha256(library) != metadata['library_sha256']:
        raise ValueError(f'GPU library hash mismatch: {library}')
    return metadata, executables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, default=REPO / 'CUDA/build')
    parser.add_argument('--short-duration', type=float, default=10, help='fm/c; reference controls')
    parser.add_argument('--long-duration', type=float, default=100, help='fm/c; parallel CPU/GPU check')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--initial-flow-atol', type=float, default=5e-6)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        for duration in (args.short_duration, args.long_duration):
            for dt in (0.1, 0.05):
                exact_steps(duration, dt)
        if args.long_duration < args.short_duration or args.threads < 1:
            raise ValueError('Long duration must be >= short duration; threads must be positive')
        if not np.isfinite(args.initial_flow_atol) or args.initial_flow_atol <= 0:
            raise ValueError('Initial flow tolerance must be finite and positive')
        metadata, executables = verify_build(args.build_dir.resolve())
        state_path = args.state.resolve()
        initial = checkpoint(state_path)
        if (initial['nucleons'] != (10, 10) or initial['states'] != 20
                or initial['force'].lower() != 'sly5'):
            raise ValueError('This first validation block requires Sly5 20Ne with 20 propagated states')
        if any(n < 2 or n % 2 for n in initial['grid']):
            raise ValueError('Even grid dimensions >= 2 are required')
        if any(not np.isfinite(d) or d <= 0 for d in initial['spacing']):
            raise ValueError('Finite positive grid spacing is required')
        if not np.allclose(initial['attributes'][0], 1, rtol=0, atol=1e-12):
            raise ValueError('This initial block requires unit occupations')
        initial_gram = gram(initial)
        del initial['psi']
    except (ValueError, OSError) as error:
        parser.error(str(error))
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix='sky3d-extended-', dir=Path.home() / '.cache'))
    seed = root / 'initial.tdhf'
    shutil.copy2(state_path, seed)
    if sha256(seed) != sha256(state_path):
        raise RuntimeError('Initial checkpoint copy changed')
    report = dict(
        run_directory=str(root), state_sha256=sha256(seed),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
        source_status=subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO, text=True).splitlines(),
        runner_sha256=sha256(Path(__file__)), build=metadata,
        nucleus='20Ne', grid=initial['grid'], spacing_fm=initial['spacing'],
        short_duration_fm_c=args.short_duration, long_duration_fm_c=args.long_duration,
        coverage='20Ne integration, unboosted and longer-run block; other nuclei/forces and production accuracy remain pending',
        limits=dict(occupied_gram_max_abs_error=1e-6, occupied_gram_max_abs_drift=1e-6,
                    checkpoint_particle_drift=1e-6, initial_flow_atol_mev=args.initial_flow_atol,
                    physical_energy_drift='recorded, not assigned a production-accuracy pass'),
        cases=[], integration_sensitivity=[], passed=False, status='running')

    def save():
        # Keep useful partial evidence if the process is interrupted or a gate fails.
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output)

    save()
    reference_paths = {}
    cases = [('boost-dt01-order4', 0.1, 4, '5.0D-5'),
             ('boost-dt005-order4', 0.05, 4, '5.0D-5'),
             ('boost-dt01-order6', 0.1, 6, '5.0D-5'),
             ('boost-dt005-order6', 0.05, 6, '5.0D-5'),
             ('unboosted-dt01-order4', 0.1, 4, '0.0D0')]
    try:
        for label, dt, order, amplitude in cases:
            text = input_text(exact_steps(args.short_duration, dt), initial['grid'], validate=True,
                              amplitude=amplitude, spacing=initial['spacing'],
                              mprint=exact_steps(1, dt), dt=dt, mxpact=order)
            row = dict(case=label, duration_fm_c=args.short_duration, runs={})
            report['cases'].append(row)
            reference = root / f'{label}-reference'
            reference_paths[label] = reference
            for backend in ('reference', 'cpu', 'gpu'):
                path = root / f'{label}-{backend}'
                result = run(executables[backend], path, text, 1 if backend == 'reference' else args.threads,
                             'cpu' if backend == 'reference' else backend, validate=backend == 'gpu')
                row['runs'][backend] = result
                save()
                result['invariants'] = invariants(path, initial_gram, dt, order)
                if backend != 'reference':
                    result['comparison'] = compare(path, reference, args.initial_flow_atol)
                save()
            row['passed'] = True
            save()
        target = reference_paths['boost-dt005-order6']
        for label in ('boost-dt01-order4', 'boost-dt005-order4', 'boost-dt01-order6'):
            report['integration_sensitivity'].append(dict(
                case=label, reference='boost-dt005-order6', backend='strict CPU',
                **integration_difference(reference_paths[label], target)))
            save()
        text = input_text(exact_steps(args.long_duration, 0.1), initial['grid'], validate=True,
                          spacing=initial['spacing'], mprint=10, dt=0.1, mxpact=4)
        row = dict(case='long-boost-dt01-order4', duration_fm_c=args.long_duration,
                   reference='optimized parallel CPU; strict CPU checked in short controls', runs={})
        report['cases'].append(row)
        for backend in ('cpu', 'gpu'):
            path = root / f'long-{backend}'
            result = run(executables[backend], path, text, args.threads, backend, validate=backend == 'gpu')
            row['runs'][backend] = result
            save()
            result['invariants'] = invariants(path, initial_gram, 0.1, 4)
            if backend == 'gpu':
                result['comparison'] = compare(path, root / 'long-cpu', args.initial_flow_atol)
            save()
        row['passed'] = True
        report.update(passed=True, status='completed')
        save()
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        raise
    print(f'Passed this validation block: {args.output}; raw output: {root}', flush=True)


if __name__ == '__main__':
    main()
