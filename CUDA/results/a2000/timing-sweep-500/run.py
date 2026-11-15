import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from benchmark import checkpoint, difference, check_observable

root = Path.cwd()
heavy = root.parent / 'heavy-6000.vj4QGl'
repo = root.parents[2]
nuclei = {'O16': (8, 8), 'Sn120': (70, 50), 'Pb208': (126, 82)}
meshes = (24, 32, 40)
static_iterations = 100
steps = 2500
dt = 0.2
settings = {
    'cpu': ('cpu', 24, None),
    'gpu-off': ('gpu', 8, '0'),
    'gpu-on': ('gpu', 8, '1'),
}
report = dict(status='running', grid_lengths=meshes, spacing_fm=1.0,
              static_iteration_cap=static_iterations, dt_fm_c=dt, steps=steps,
              duration_fm_c=500, mprint_dynamic=50, settings=settings, cases={})

def save():
    tmp = root / 'summary.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    tmp.replace(root / 'summary.json')

def replace(text, key, value):
    text, count = re.subn(r'(?i)\b' + key + r'\s*=\s*[^,\s/]+', f'{key}={value}', text)
    if count != 1:
        raise ValueError(f'Expected one namelist value for {key}, found {count}')
    return text

def mesh(text, length):
    for key in ('nx', 'ny', 'nz'):
        text = replace(text, key, length)
    for key in ('dx', 'dy', 'dz'):
        text = replace(text, key, '1.0')
    return text

def load_input(nucleus, length, phase):
    if nucleus == 'O16':
        text = (repo / 'Test/GPU' / ('for005.static' if phase == 'static' else 'for005.tdhf')).read_text()
    else:
        text = (heavy / nucleus / ('static' if phase == 'static' else 'pilot-gpu') / 'for005').read_text()
    text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('!')) + '\n'
    values = (dict(mprint=20, mplot=0, mrest=static_iterations, maxiter=static_iterations, writeselect="'r'")
              if phase == 'static' else dict(mprint=50, mplot=0, mrest=steps, nt=steps, dt=dt, writeselect="'r'"))
    for key, value in values.items():
        text = replace(text,key,value)
    return mesh(text, length)

def run(case, phase, variant, text, seed=None):
    folder = root / 'runs' / case / phase / variant
    folder.mkdir(parents=True)
    (folder/'for005').write_text(text)
    if seed is not None:
        (folder/'initial.tdhf').symlink_to(seed)
    backend, threads, cache = settings[variant]
    env = {k:v for k,v in os.environ.items() if not k.startswith('SKY3D_') and not k.startswith('TRIAL_')}
    env.update(SKY3D_BACKEND=backend, SKY3D_GPU_GRAPHS='1', SKY3D_GPU_MEMORY_FRACTION='0.8',
               OMP_NUM_THREADS=str(threads), OMP_PLACES='cores', OMP_PROC_BIND='close',
               OPENBLAS_NUM_THREADS='1')
    if cache is not None:
        env['TRIAL_STAGE_CACHE'] = cache
    row = report['cases'][case][phase][variant] = dict(status='running', path=str(folder.relative_to(root)))
    print(f'Starting {case} {phase} {variant}', flush=True)
    save()
    try:
        with (folder/'stdout.log').open('w') as out:
            start = time.perf_counter()
            result = subprocess.run([str(root/backend/f'sky3d.{backend}')], cwd=folder, env=env,
                                    stdout=out, stderr=subprocess.STDOUT)
            row['seconds'] = time.perf_counter() - start
        if result.returncode:
            raise RuntimeError(f'Exit status {result.returncode}')
        log = (folder/'stdout.log').read_text()
        if backend=='gpu':
            mem = re.search(r'GPU memory: required ([\d.]+) MiB .*; free ([\d.]+) MiB',log)
            if not mem:
                raise RuntimeError('GPU memory report missing')
            row.update(required_mib=float(mem[1]),free_mib=float(mem[2]))
        if phase == 'static':
            match = re.search(r'Static loop wall seconds: ([^;]+); iterations: (\d+)',log)
            if not match:
                raise RuntimeError('Static loop wall timer missing')
            fluct = re.search(r'Final pre-gradient fluctuations \(MeV\): h\*\*2=\s*([^,]+)',log)
            row.update(loop_seconds=float(match[1]),iterations=int(match[2]),
                       converged='Static convergence criterion met' in log,
                       fluctuation_mev=float(fluct[1]))
            statepath = folder/'initial.tdhf'
        else:
            statepath = folder/'dynamic.tdhf'
        state = checkpoint(statepath)
        nucleus,length = case.split('-')
        if state['grid']!=(int(length),)*3 or state['nucleons']!=nuclei[nucleus]:
            raise RuntimeError('Unexpected grid or nucleus')
        if phase == 'dynamic':
            if state['step']!=steps or abs(state['time']-500)>1e-7:
                raise RuntimeError(f"Missing endpoint, {state['step']=} {state['time']=}")
            row.update(final_time=state['time'])
        row.update(status='complete', states=state['states'], final_particles=state['particles'])
        print(f"Completed {case} {phase} {variant}: {row['seconds']:.3f}s",flush=True)
        return statepath
    except Exception as exc:
        row.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        print(f"Failed {case} {phase} {variant}: {row['error']}",flush=True)
        return None
    finally:
        save()

def compare(case, files):
    if not files.get('cpu'):
        return
    cpu = checkpoint(files['cpu'])
    for variant, path in files.items():
        if variant=='cpu' or path is None:
            continue
        row = report['cases'][case]['dynamic'][variant]
        gpu = checkpoint(path)
        wave = difference(gpu['psi'], cpu['psi'])
        row['cpu_comparison'] = {'wavefunction':wave, 'wavefunction_passed':max(wave.values())<1e-9,
                                 'particle_max_abs_difference':max(abs(a-b) for a,b in zip(cpu['particles'],gpu['particles']))}
        checks = row['cpu_comparison']['observables'] = {}
        for name in ('energies.res','quadrupoles.res','monopoles.res'):
            a,b = (np.loadtxt(p.parent/name,ndmin=2) for p in (path,files['cpu']))
            try:
                check_observable(name,a,b)
            except (ValueError,AssertionError):
                checks[name] = False
            else:
                checks[name] = True
        save()

def run_case(nucleus,length,order):
    case = f'{nucleus}-{length}'
    report['cases'][case] = dict(static={}, dynamic={})
    static = {}
    text=load_input(nucleus,length,'static')
    for variant in ('cpu','gpu-off'):
        static[variant] = run(case,'static',variant,text)
    if not static['cpu']:
        return
    with static['cpu'].open('rb') as stream:
        report['cases'][case]['shared_static_sha256'] = hashlib.file_digest(stream,'sha256').hexdigest()
    save()
    text=load_input(nucleus,length,'dynamic')
    files={variant:run(case,'dynamic',variant,text,static['cpu']) for variant in order}
    compare(case,files)

save()
try:
    for n,nucleus in enumerate(nuclei):
        for j,length in enumerate(meshes):
            order = ('cpu','gpu-off','gpu-on')
            if (n+j)%2:
                order=order[::-1]
            run_case(nucleus,length,order)
    expected=len(meshes)*len(nuclei)*5
    complete=sum(row['status']=='complete' for c in report['cases'].values()
                 for phase in ('static','dynamic') for row in c[phase].values())
    report.update(status='complete' if complete==expected else 'incomplete',
                  completed_runs=complete,expected_runs=expected)
    save()
except Exception as e:
    report.update(status='failed',error=f'{type(e).__name__}: {e}')
    save()
    raise

