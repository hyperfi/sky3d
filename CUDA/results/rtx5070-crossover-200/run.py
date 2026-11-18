"""Run the A2000 input matrix at 200 fm/c on the local checkout."""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import struct
import subprocess
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from benchmark import REPO, check_observable, checkpoint, record
from validate_extended import sha256, verify_build

FIXTURES = REPO / 'CUDA/results/a2000/timing-sweep-500/runs'
NUCLEI = {'Pb208': (126, 82), 'Sn120': (70, 50), 'O16': (8, 8)}
GRIDS = (24, 32, 36, 40)
THREADS = (4, 8, 12, 16, 20)
PILOT_STEPS = 50
STEPS = 1000
DT = 0.2
INVARIANT_ATOL = 1e-6


def set_value(text, key, value):
    text, count = re.subn(rf'(?i)\b{key}\s*=\s*[^,\s/]+', f'{key}={value}', text)
    if count != 1:
        raise ValueError(f'{key} must occur once in the archived input')
    return text


def gram_error(state, path):
    with path.open('rb') as f:
        split = struct.unpack_from('<i', record(f), 40)[0]
    psi = state['psi']
    return float(max(np.max(np.abs(np.prod(state['spacing']) * p.conj() @ p.T - np.eye(len(p))))
                     for p in (psi[:split], psi[split:])))


def wave_error(actual, expected):
    # Compare one orbital at a time to limit peak memory on the 40^3 grids.
    if actual.shape != expected.shape:
        raise ValueError('Endpoint orbital arrays have different shapes')
    diff_norm, ref_norm, max_abs = 0.0, 0.0, 0.0
    for a, b in zip(actual, expected):
        delta = a - b
        diff_norm += np.vdot(delta, delta).real
        ref_norm += np.vdot(b, b).real
        max_abs = max(max_abs, np.max(np.abs(delta)))
    return dict(relative_l2=float(np.sqrt(diff_norm / max(ref_norm, 1e-300))), max_abs=float(max_abs))


def load_histories(path, steps):
    expected_times = np.arange(0, steps + 1, PILOT_STEPS) * DT
    histories = {}
    for name in ('energies.res', 'quadrupoles.res', 'monopoles.res'):
        values = np.loadtxt(path / name, ndmin=2)
        if (not np.isfinite(values).all() or len(values) != len(expected_times) or
                not np.allclose(values[:, 0], expected_times, atol=1e-7, rtol=0)):
            raise ValueError(f'Incomplete, unmatched or nonfinite samples in {name}')
        histories[name] = values
    return histories


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workspace', type=Path, help='Empty Linux filesystem directory for raw runs')
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    build = args.build.resolve()
    meta, executables = verify_build(build)
    work = args.workspace.resolve()
    if any(work.iterdir()):
        raise ValueError(f'Raw run directory must be empty: {work}')
    if args.report.exists():
        raise FileExistsError(args.report)

    report = dict(status='running', raw_directory=str(work), build=meta,
                  runner_sha256=sha256(Path(__file__)),
                  hardware=dict(cpu=next(line.split(':', 1)[1].strip() for line in
                                         Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),
                                gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                                                             '--format=csv,noheader'], text=True).strip(),
                                kernel=os.uname().release),
                  dt_fm_c=DT, duration_fm_c=STEPS * DT, steps=STEPS, spacing_fm=1.0,
                  cpu_pilot_threads=THREADS, pilot_steps=PILOT_STEPS, gpu_host_threads=8,
                  static_preparation='Same A2000 archive input, up to 100 static iterations. Shared CPU-prepared checkpoint per pair.',
                  timings='Sequential complete jobs, one 200 fm/c pair per case, alternating backend order by case. Pilot chooses fastest tested OpenMP configuration.',
                  omp_places='cores', omp_proc_bind='close', openblas_threads=1,
                  cases={})

    def save():
        tmp = args.report.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        tmp.replace(args.report)

    def run(folder, text, backend, threads, entry, seed=None, steps=None):
        folder.mkdir(parents=True)
        (folder / 'for005').write_text(text)
        if seed is not None:
            (folder / 'initial.tdhf').symlink_to(seed)
        env = {k: v for k, v in os.environ.items() if not k.startswith(('SKY3D_', 'TRIAL_', 'OMP_'))}
        env.update(SKY3D_BACKEND=backend, SKY3D_GPU_GRAPHS='1', OPENBLAS_NUM_THREADS='1',
                   OMP_NUM_THREADS=str(threads), OMP_PROC_BIND='close', OMP_PLACES='cores',
                   LD_LIBRARY_PATH=str(build))
        entry.update(status='running', directory=str(folder), input_sha256=sha256(folder / 'for005'), threads=threads)
        report['active'] = str(folder.relative_to(work))
        save()
        print('Starting', report['active'], flush=True)
        try:
            with (folder / 'stdout.log').open('w') as log:
                start = time.perf_counter()
                process = subprocess.run([str(executables[backend])], cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
                entry['seconds'] = time.perf_counter() - start
            entry['returncode'] = process.returncode
            if process.returncode:
                raise ValueError(f'Sky3D returned {process.returncode}')
            path = folder / ('initial.tdhf' if steps is None else 'dynamic.tdhf')
            state = checkpoint(path)
            name, grid = folder.relative_to(work).parts[0].split('-')
            if state['grid'] != (int(grid),)*3 or state['nucleons'] != NUCLEI[name]:
                raise ValueError('Wrong saved nucleus or grid')
            log = (folder / 'stdout.log').read_text(errors='replace')
            if backend == 'gpu':
                memory = re.search(r'GPU memory: required ([\d.]+) MiB .*; free ([\d.]+) MiB', log)
                if not memory:
                    raise ValueError('GPU memory report missing')
                entry.update(required_mib=float(memory[1]), free_mib=float(memory[2]))
            entry.update(states=state['states'], step=state['step'], time_fm_c=state['time'])
            if steps is None:
                entry['converged'] = 'Static convergence criterion met' in log
                if state['step'] != 100 and not entry['converged']:
                    raise ValueError('Static seed stopped before the iteration cap')
            else:
                if state['step'] != steps or abs(state['time'] - DT * steps) > 1e-7:
                    raise ValueError('Saved endpoint did not reach the requested duration')
                histories = load_histories(folder, steps)
                if steps == STEPS:
                    energy = histories['energies.res'][:, 4]
                    entry.update(final_particles=state['particles'], endpoint_gram_error=gram_error(state, path),
                                 integrated_energy_max_drift_mev=float(np.max(np.abs(energy-energy[0]))),
                                 checkpoint_sha256=sha256(path))
            entry['status'] = 'complete'
            print(f"Completed {report['active']} in {entry['seconds']:.3f}s", flush=True)
            return path
        except (ValueError, RuntimeError, OSError, struct.error, TypeError) as error:
            entry.update(status='failed', error=str(error))
            print('Failed', report['active'], error, flush=True)
            return None
        finally:
            save()

    def campaign():
        for grid_index, grid in enumerate(GRIDS):
            for nucleus_index, nucleus in enumerate(NUCLEI):
                name = f'{nucleus}-{grid}'
                static = (FIXTURES / name / 'static/cpu/for005').read_text()
                dynamic = (FIXTURES / name / 'dynamic/cpu/for005').read_text()
                case = report['cases'][name] = dict(static={}, cpu_pilots={}, dynamic={})
                seed = run(work / name / 'static', static, 'cpu', 8, case['static'])
                if seed is None:
                    continue
                case['shared_seed_sha256'] = sha256(seed)
                case['static']['gram_error'] = gram_error(checkpoint(seed), seed)
                pilot = set_value(set_value(dynamic, 'nt', PILOT_STEPS), 'mrest', PILOT_STEPS)
                order = THREADS if (grid_index + nucleus_index) % 2 == 0 else THREADS[::-1]
                for threads in order:
                    run(work / name / 'pilot' / f'cpu-{threads}', pilot, 'cpu', threads,
                        case['cpu_pilots'].setdefault(str(threads), {}), seed, PILOT_STEPS)
                speeds = {int(k): p['seconds'] for k, p in case['cpu_pilots'].items() if p['status'] == 'complete'}
                if not speeds:
                    continue
                fastest = sorted(speeds, key=speeds.get)[:2]
                if len(fastest) == 2 and speeds[fastest[1]] / speeds[fastest[0]] < 1.10:
                    for threads in reversed(fastest):
                        row = case['cpu_pilots'][str(threads)]
                        run(work / name / 'pilot' / f'retest-cpu-{threads}', pilot, 'cpu', threads,
                            row.setdefault('retest', {}), seed, PILOT_STEPS)
                        if row['retest']['status'] == 'complete':
                            speeds[threads] = statistics.mean((speeds[threads], row['retest']['seconds']))
                    selected = min(fastest, key=speeds.get)
                else:
                    selected = fastest[0]
                case['selected_cpu_threads'] = selected
                full = set_value(set_value(dynamic, 'nt', STEPS), 'mrest', STEPS)
                order = ('cpu', 'gpu') if (grid_index + nucleus_index) % 2 == 0 else ('gpu', 'cpu')
                paths = {backend: run(work / name / backend, full, backend, selected if backend == 'cpu' else 8,
                                      case['dynamic'].setdefault(backend, {}), seed, STEPS) for backend in order}
                if not all(paths.values()):
                    continue
                cpu, gpu = (checkpoint(paths[b]) for b in ('cpu', 'gpu'))
                compare = case['comparison'] = dict(wavefunction=wave_error(gpu['psi'], cpu['psi']), observables={})
                del cpu, gpu
                cpu_hist, gpu_hist = (load_histories(paths[b].parent, STEPS) for b in ('cpu', 'gpu'))
                for name_res, reference in cpu_hist.items():
                    actual = gpu_hist[name_res]
                    same = True
                    try:
                        check_observable(name_res, actual, reference)
                    except AssertionError:
                        same = False
                    compare['observables'][name_res] = dict(passed=same, max_abs_by_column=np.max(np.abs(actual-reference), axis=0).tolist())
                compare['equivalence_passed'] = max(compare['wavefunction'].values()) <= 1e-9 and all(
                    row['passed'] for row in compare['observables'].values())
                for row in case['dynamic'].values():
                    row['particle_max_drift'] = max(abs(p - n) for p, n in zip(row['final_particles'], NUCLEI[nucleus]))
                    row['endpoint_invariants_passed'] = max(row['endpoint_gram_error'], row['particle_max_drift']) <= INVARIANT_ATOL
                case['cpu_over_gpu'] = case['dynamic']['cpu']['seconds'] / case['dynamic']['gpu']['seconds']
                print(f"Pair {name}: CPU/GPU {case['cpu_over_gpu']:.3f}, agreement {compare['equivalence_passed']}", flush=True)
                save()

    try:
        campaign()
        report['status'] = 'complete' if all('comparison' in c for c in report['cases'].values()) else 'incomplete'
    except BaseException as error:
        report.update(status='interrupted', error=str(error))
        raise
    finally:
        report.pop('active', None)
        save()


if __name__ == '__main__':
    main()
