#!/usr/bin/env python3
"""Measure 20Ne overlap drift without discarding failed numerical controls.

Runs use one immutable checkpoint and matched physical output times. Raw
checkpoints and logs remain in the WSL cache; only compact evidence is exported.
An invariant pass is not a claim of spatial or response convergence.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from benchmark import REPO, checkpoint, difference, input_text
from validate_extended import exact_steps, gram, sha256, verify_build


def overlap_diagnostics(state, initial):
    matrices = gram(state)
    if not all(np.isfinite(g).all() for g in matrices):
        raise ValueError('Nonfinite overlap matrix')
    diagonal = max(np.max(np.abs(np.diag(g) - 1)) for g in matrices)
    off_diagonal = max(np.max(np.abs(g - np.diag(np.diag(g)))) for g in matrices)
    drift = max(np.max(np.abs(g - g0)) for g, g0 in zip(matrices, initial))
    error = max(diagonal, off_diagonal)
    particles = float(np.max(np.abs(np.asarray(state['particles']) - 10)))
    return dict(gram_max_abs_error=float(error), gram_max_abs_drift=float(drift),
                norm_max_abs_error=float(diagonal), overlap_max_abs=float(off_diagonal),
                particles=state['particles'], particle_max_abs_drift=particles,
                invariant_passed=bool(error <= 1e-6 and drift <= 1e-6 and particles <= 1e-6))


def history(path, duration):
    values = np.loadtxt(path, ndmin=2)
    if not np.isfinite(values).all():
        raise ValueError(f'Nonfinite history: {path}')
    if abs(values[0, 0]) > 1e-8 or abs(values[-1, 0] - duration) > 1e-8:
        raise ValueError(f'Missing exact [0,{duration}] history: {path}')
    return values


def matched_delta(boost, unboosted):
    if boost.shape != unboosted.shape or not np.allclose(boost[:, 0], unboosted[:, 0], rtol=0, atol=1e-8):
        raise ValueError('Boost and unboosted histories require identical physical sample times')
    delta = boost[:, 1] - unboosted[:, 1]
    return delta - delta[0]


def execute(exe, path, text, backend, threads, duration, steps, initial):
    path.mkdir()
    (path / 'for005').write_text(text)
    env = dict(os.environ, SKY3D_BACKEND=backend, OMP_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS='1', OMP_PROC_BIND='close', OMP_PLACES='cores')
    env.pop('SKY3D_GPU_VALIDATE', None)
    start = time.perf_counter()
    with (path / 'stdout.log').open('w') as log:
        process = subprocess.run([str(exe)], cwd=path, env=env, stdout=log, stderr=subprocess.STDOUT)
    result = dict(directory=str(path), seconds=time.perf_counter()-start,
                  returncode=process.returncode, input_sha256=sha256(path / 'for005'),
                  completed=False, invariant_passed=False)
    try:
        if process.returncode:
            raise ValueError(f'Executable returned {process.returncode}')
        state = checkpoint(path / 'k0_restart.tdhf')
        result.update(final_step=state['step'], final_time_fm_c=state['time'],
                      checkpoint_sha256=sha256(path / 'k0_restart.tdhf'))
        # Legacy STOP can return zero. A checkpoint alone is not completion.
        if state['step'] != steps or abs(state['time'] - duration) > 1e-8:
            raise ValueError('Incomplete checkpoint endpoint (including possible legacy STOP)')
        energy = history(path / 'energies.res', duration)
        quadrupole = history(path / 'quadrupoles.res', duration)
        result.update(overlap_diagnostics(state, initial))
        result.update(completed=True,
                      integrated_energy_max_drift_mev=float(np.max(np.abs(energy[:, 4]-energy[0, 4]))),
                      uncorrected_quadrupole_max_drift_fm2=float(np.max(np.abs(quadrupole[:, 1]-quadrupole[0, 1]))),
                      observable_sha256={name: sha256(path / name) for name in ('energies.res', 'quadrupoles.res')})
        np.savez(path / 'occupied_gram.npz', neutron=gram(state)[0], proton=gram(state)[1])
    except (ValueError, OSError) as error:
        result['failure'] = str(error)
        result['stdout_tail'] = (path / 'stdout.log').read_text(errors='replace')[-2000:]
    return result


def summarize(report):
    complete = [r for r in report['cases'] if r['completed']]
    responses = {}
    for row in complete:
        if row['boosted']:
            baseline = next((r for r in complete if not r['boosted'] and
                             all(r[k] == row[k] for k in ('dt_fm_c', 'taylor_order', 'backend'))), None)
            if baseline:
                q = history(Path(row['directory']) / 'quadrupoles.res', report['duration_fm_c'])
                u = history(Path(baseline['directory']) / 'quadrupoles.res', report['duration_fm_c'])
                responses[row['label']] = (row, q[:, 0], matched_delta(q, u))
    report['integration_sensitivity'] = []
    for backend in report['backends']:
        candidates = [v for v in responses.values() if v[0]['backend'] == backend]
        if not candidates:
            continue
        reference = min(candidates, key=lambda v: (v[0]['dt_fm_c'], -v[0]['taylor_order']))
        for row, times, delta in candidates:
            if not np.allclose(times, reference[1], rtol=0, atol=1e-8):
                raise ValueError('Integration sensitivity requires matched output times')
            report['integration_sensitivity'].append(dict(case=row['label'], reference=reference[0]['label'],
                reference_is_converged=False, matched_unboosted_delta=difference(delta, reference[2])))
    report['cpu_gpu_equivalence'] = []
    for gpu in complete:
        if gpu['backend'] != 'gpu':
            continue
        cpu = next((r for r in complete if r['backend'] == 'cpu' and
                    all(r[k] == gpu[k] for k in ('dt_fm_c', 'taylor_order', 'boosted'))), None)
        if cpu:
            a, b = [checkpoint(Path(r['directory']) / 'k0_restart.tdhf') for r in (gpu, cpu)]
            wave = difference(a['psi'], b['psi'])
            overlap = float(max(np.max(np.abs(x-y)) for x, y in zip(gram(a), gram(b))))
            report['cpu_gpu_equivalence'].append(dict(case=gpu['label'], wavefunction=wave,
                gram_max_abs_difference=overlap, passed=bool(wave['relative_l2'] < 1e-9 and wave['max_abs'] < 1e-9 and overlap < 1e-9)))
    return responses


def plot(report, output, responses):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), constrained_layout=True)
    for order in sorted({r['taylor_order'] for r in report['cases']}):
        for boosted in (False, True):
            rows = sorted([r for r in report['cases'] if r['completed'] and r['backend'] == 'gpu'
                           and r['taylor_order'] == order and r['boosted'] == boosted], key=lambda r:r['dt_fm_c'])
            if rows:
                axes[0].loglog([r['dt_fm_c'] for r in rows], [r['gram_max_abs_error'] for r in rows],
                               marker='o', ls='-' if boosted else '--',
                               label=f"Order {order}, {'boosted' if boosted else 'unboosted'}")
    axes[0].axhline(1e-6, color='black', ls=':', label='Overlap limit')
    axes[0].set(xlabel='Timestep (fm/c)', ylabel='Maximum |G − I|', title='Endpoint norm and overlap error')
    for label, (row, times, delta) in responses.items():
        if row['backend'] == 'gpu':
            axes[1].plot(times, delta, lw=1.1, label=f"dt={row['dt_fm_c']:g}, order={row['taylor_order']}")
    axes[1].set(xlabel='Time (fm/c)', ylabel=r'$\delta\langle F_{20}\rangle$ (fm$^2$)',
                title='Quadrupole response with matched unboosted subtraction')
    for axis in axes:
        axis.legend(frameon=False, fontsize=8)
        axis.grid(alpha=0.18)
    fig.suptitle(f"SLy5 20Ne · {report['grid']} · {report['duration_fm_c']:g} fm/c")
    fig.savefig(output.with_suffix('.png'), dpi=170)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=200)
    parser.add_argument('--sample-interval', type=float, default=2)
    parser.add_argument('--dt', type=float, nargs='+', default=[0.2, 0.1, 0.05])
    parser.add_argument('--order', type=int, nargs='+', default=[4, 6])
    parser.add_argument('--backends', choices=['cpu', 'gpu'], nargs='+', default=['gpu'])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    metadata, executables = verify_build(args.build_dir.resolve())
    state_path = args.state.resolve()
    initial = checkpoint(state_path)
    if (initial['nucleons'] != (10, 10) or initial['states'] != 20 or initial['force'].lower() != 'sly5'
            or not np.allclose(initial['attributes'][0], 1, rtol=0, atol=1e-12)):
        parser.error('Requires SLy5 20Ne with ten unit-occupied orbitals per species')
    if args.threads < 1 or min(args.order) < 2:
        parser.error('Positive threads and Taylor orders >= 2 required')
    for dt in args.dt:
        exact_steps(args.duration, dt)
        exact_steps(args.sample_interval, dt)
    exact_steps(args.duration, args.sample_interval)
    root = Path(tempfile.mkdtemp(prefix='sky3d-overlap-', dir=Path.home()/'.cache'))
    shutil.copy2(state_path, root / 'initial.tdhf')
    if sha256(root / 'initial.tdhf') != sha256(state_path):
        raise ValueError('Initial checkpoint copy changed')
    initial_gram = gram(initial)
    report = dict(run_directory=str(root), state_path=str(state_path), state_sha256=sha256(state_path),
                  build=metadata, runner_sha256=sha256(Path(__file__)),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
                  grid=initial['grid'], spacing_fm=initial['spacing'], duration_fm_c=args.duration,
                  sample_interval_fm_c=args.sample_interval, backends=args.backends, threads=args.threads,
                  limits=dict(gram_max_abs_error=1e-6, gram_max_abs_drift=1e-6, particle_max_abs_drift=1e-6),
                  interpretation='Fixed-state timestep/order sensitivity; no automatic spatial or spectral convergence claim',
                  cases=[], status='running')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)
    save()
    try:
        for dt in args.dt:
            for order in args.order:
                for boosted in (False, True):
                    text = input_text(exact_steps(args.duration, dt), initial['grid'],
                                      amplitude='5.0D-5' if boosted else '0.0D0', spacing=initial['spacing'],
                                      mprint=exact_steps(args.sample_interval, dt), dt=dt, mxpact=order)
                    for backend in args.backends:
                        label = f"{'boost' if boosted else 'unboosted'}-dt{dt:g}-order{order}-{backend}"
                        row = execute(executables[backend], root/label, text, backend, args.threads,
                                      args.duration, exact_steps(args.duration, dt), initial_gram)
                        row.update(label=label, dt_fm_c=dt, taylor_order=order, backend=backend, boosted=boosted)
                        report['cases'].append(row)
                        save()
                        print(json.dumps({k:row.get(k) for k in ('label','seconds','completed','gram_max_abs_error','invariant_passed','failure')}), flush=True)
        responses = summarize(report)
        report['status'] = 'complete'
        report['all_cases_invariant_passed'] = all(r['completed'] and r['invariant_passed'] for r in report['cases'])
        save()
        plot(report, args.output, responses)
    except BaseException:
        report['status'] = 'interrupted_or_analysis_failed'
        save()
        raise
    print(f'Evidence: {args.output}', flush=True)


if __name__ == '__main__':
    main()
