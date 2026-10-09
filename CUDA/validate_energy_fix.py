#!/usr/bin/env python3
"""Check diagnostic freshness, legacy trajectories, external pulses and restarts."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from benchmark import REPO, checkpoint, compare, difference, input_text
from check_energy_timing import correction
from validate_extended import sha256, verify_build


def execute(exe, path, text, threads, backend, restart=None):
    path.mkdir()
    (path / 'for005').write_text(text)
    if restart:
        shutil.copy2(restart, path / 'k0_restart.tdhf')
    env = dict(os.environ, SKY3D_BACKEND=backend, OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1', OMP_PROC_BIND='close', OMP_PLACES='cores')
    env.pop('SKY3D_GPU_VALIDATE', None)
    start = time.perf_counter()
    with (path / 'stdout.log').open('w') as log:
        subprocess.run([str(exe)], cwd=path, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
    state = checkpoint(path / 'k0_restart.tdhf')
    steps = int(re.search(r'\bnt=(\d+)', text).group(1))
    dt = float(re.search(r'\bdt=([\d.eE+-]+)', text).group(1))
    if state['step'] != steps or abs(state['time'] - steps*dt) > 1e-8:
        raise AssertionError(f'Missing endpoint: {path}')
    if max(abs(p - 10) for p in state['particles']) > 1e-6:
        raise AssertionError(f'Particle drift: {path}')
    return dict(directory=str(path), seconds=time.perf_counter()-start,
                input_sha256=sha256(path / 'for005'), particles=state['particles'])


def wave_check(actual, reference):
    a, b = checkpoint(actual / 'k0_restart.tdhf'), checkpoint(reference / 'k0_restart.tdhf')
    if a['step'] != b['step'] or abs(a['time'] - b['time']) > 1e-8:
        raise AssertionError('Mismatched endpoint')
    error = difference(a['psi'], b['psi'])
    if error['relative_l2'] > 1e-9 or error['max_abs'] > 1e-9:
        raise AssertionError(error)
    return error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--legacy-build', type=Path, default=REPO / 'CUDA/build')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-passed', type=Path, help='Reuse passed cases from a matching build/state report')
    args = parser.parse_args()
    build, executables = verify_build(args.build_dir.resolve())
    legacy = args.legacy_build.resolve()
    old = json.loads((legacy / 'build.json').read_text())
    legacy_exe = legacy / 'reference/sky3d.reference'
    if sha256(legacy_exe) != old['executable_sha256']['reference']:
        parser.error('Legacy reference executable hash mismatch')
    state = checkpoint(args.state)
    if state['nucleons'] != (10, 10) or state['states'] != 20 or state['force'].lower() != 'sly5':
        parser.error('Sly5 20Ne / 20-state checkpoint required')
    root = Path(tempfile.mkdtemp(prefix='sky3d-energy-fix-', dir=Path.home() / '.cache'))
    shutil.copy2(args.state, root / 'initial.tdhf')
    report = dict(run_directory=str(root), state_sha256=sha256(root / 'initial.tdhf'),
                  build=build, legacy_build=old, runner_sha256=sha256(Path(__file__)), cases=[], passed=False)
    reused = {}
    if args.reuse_passed:
        previous = json.loads(args.reuse_passed.read_text())
        if (previous['state_sha256'] != report['state_sha256'] or
                previous['build']['executable_sha256'] != build['executable_sha256'] or
                previous['legacy_build']['executable_sha256'] != old['executable_sha256']):
            parser.error('Cannot reuse checks from a different state/build')
        reused = {row['case']: row for row in previous['cases'] if row.get('passed')}
        shutil.copy2(args.reuse_passed, Path(previous['run_directory']) / 'original-report.json')
        report['reuse_note'] = 'Passed cases reused with matching state/executable hashes and identical input text; failed pulse fixture retained in original cache'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    def fresh_check(path):
        data = checkpoint(path / 'k0_restart.tdhf')
        result = correction(sorted(path.glob('*.tdd'))[-1], data)
        if result['potential_max_abs_difference_mev'] > 1e-12:
            raise AssertionError(f'Endpoint Coulomb field is stale: {result}')
        return result

    def case_text(steps, mprint=10, pulse=False, restart=False):
        text = input_text(steps, state['grid'], validate=True, spacing=state['spacing'],
                          mprint=mprint, dt=0.1)
        if pulse:
            text = text.replace('ipulse=0,', 'ipulse=1, tau0=2.5, taut=0.5, omega=0.7,')
        if restart:
            text = text.replace('imode=2,', 'imode=2, trestart=T,')
        return text

    try:
        for label, steps, pulse in [('kick', 100, False), ('pulse', 40, True)]:
            text = case_text(steps, mprint=10 if not pulse else 2, pulse=pulse)
            if label in reused:
                row = reused[label]
                for data in row['runs'].values():
                    if (Path(data['directory']) / 'for005').read_text() != text:
                        raise ValueError('Reused case input differs')
                report['cases'].append(row)
                save()
                continue
            row = dict(case=label, runs={})
            report['cases'].append(row)
            reference = root / f'{label}-reference'
            for name, exe, threads in [('legacy', legacy_exe, 1),
                                       ('reference', executables['reference'], 1),
                                       ('cpu', executables['cpu'], 8), ('gpu', executables['gpu'], 8)]:
                path = root / f'{label}-{name}'
                row['runs'][name] = execute(exe, path, text, threads, 'gpu' if name == 'gpu' else 'cpu')
                if name != 'legacy':
                    row['runs'][name]['fresh_coulomb'] = fresh_check(path)
                if name in ('cpu', 'gpu'):
                    row['runs'][name]['comparison'] = compare(path, reference, 1e-5)
                save()
            row['legacy_wavefunction_comparison'] = wave_check(reference, root / f'{label}-legacy')
            for file in ('quadrupoles.res', 'monopoles.res'):
                a, b = [np.loadtxt(root / f'{label}-{name}' / file, ndmin=2) for name in ('reference', 'legacy')]
                if not np.allclose(a, b, rtol=0, atol=1e-9):
                    raise AssertionError(f'Changed trajectory: {label}: {file}')
            row['passed'] = True
            save()
        # Output schedules cannot alter dynamics, even with evolving external pulses.
        for mprint in (0, 1, 7):
            row = dict(case=f'pulse-mprint-{mprint}', runs={})
            report['cases'].append(row)
            for name in ('cpu', 'gpu'):
                path = root / f'pulse-mprint-{mprint}-{name}'
                row['runs'][name] = execute(executables[name], path, case_text(40, mprint, True), 8, name)
                row['runs'][name]['wavefunction'] = wave_check(path, root / 'pulse-reference')
                row['runs'][name]['fresh_coulomb'] = fresh_check(path)
            row['passed'] = True
            save()
        # Resume a nonzero-time external pulse at two boundaries.
        seed = root / 'pulse-seed'
        execute(executables['reference'], seed, case_text(20, 2, True), 1, 'cpu')
        for final in (30, 40):
            row = dict(case=f'pulse-restart-to-{final}', runs={})
            report['cases'].append(row)
            for name, threads in [('reference', 1), ('cpu', 8), ('gpu', 8)]:
                path = root / f'pulse-restart-{final}-{name}'
                row['runs'][name] = execute(executables[name], path, case_text(final, 2, True, True),
                                           threads, 'cpu' if name == 'reference' else name,
                                           restart=seed / 'k0_restart.tdhf')
                if final == 40:
                    row['runs'][name]['wavefunction'] = wave_check(path, root / 'pulse-reference')
                row['runs'][name]['fresh_coulomb'] = fresh_check(path)
                save()
            if final == 30:
                seed = root / 'pulse-restart-30-reference'
            row['passed'] = True
            save()
        report['passed'] = True
        save()
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    print(f'Energy-fix checks passed: {args.output}; raw output: {root}', flush=True)


if __name__ == '__main__':
    main()
