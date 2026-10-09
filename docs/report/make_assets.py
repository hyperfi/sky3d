#!/usr/bin/env python3
"""Generate report figures/tables from immutable evidence; never run Sky3D."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FINAL = REPO / 'CUDA/results/20ne-40x40x40/single-gpu'
QUAD = REPO / 'CUDA/results/20ne-k0-final'
ASSETS = HERE / 'assets'
TABLES = HERE / 'tables'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.spines.top': False,
    'axes.spines.right': False, 'axes.titleweight': 'bold',
    'savefig.facecolor': 'white'})
BLUE, ORANGE, GREY = '#205593', '#c05e1c', '#53616d'
inputs = {}

def load(path):
    path = Path(path)
    data = path.read_bytes()
    inputs[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
    return json.loads(data)

def final(name):
    return load(FINAL / (name + '.json'))

def save(fig, name):
    fig.savefig(ASSETS / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(ASSETS / (name + '.png'), bbox_inches='tight', dpi=160)
    plt.close(fig)

def tex_table(name, cols, rows):
    lines = [r'\begin{tabular}{' + cols + '}', r'\toprule']
    lines.extend(' & '.join(row) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (TABLES / (name + '.tex')).write_text('\n'.join(lines) + '\n')

def sci(value):
    m, e = f'{value:.4e}'.split('e')
    return r'$' + m + r'\times10^{' + str(int(e)) + '}$'

def main():
    ASSETS.mkdir(exist_ok=True)
    TABLES.mkdir(exist_ok=True)
    b = final('benchmark-final4')
    configs = ['cpu-4', 'cpu-8', 'cpu-20', 'gpu-8']
    rows = [[r'Configuration', r'Trial 1 (s)', r'Trial 2 (s)', r'Trial 3 (s)', r'Median (s)'], [r'\midrule', '', '', '', '']]
    # A rule occupies its own line, rather than a data row.
    rows.pop()
    samples = []
    for key in configs:
        backend, threads = key.split('-')
        values = [r['seconds'] for r in b['runs'] if r['backend'] == backend and r['threads'] == int(threads)]
        samples.append(values)
        rows.append([key.upper()] + [f'{v:.6f}' for v in values] + [f'{np.median(values):.6f}'])
    tex_table('benchmark', 'lrrrr', rows)
    fig, ax = plt.subplots(figsize=(7.1, 3.9), constrained_layout=True)
    med = [np.median(v) for v in samples]
    ax.bar(range(4), med, color=[BLUE, BLUE, BLUE, ORANGE], width=.58, alpha=.86)
    for i, values in enumerate(samples):
        ax.scatter(i + np.linspace(-.11, .11, len(values)), values, color='black', s=20, zorder=3)
        ax.text(i, med[i] + 4, f'{med[i]:.2f} s', ha='center', fontsize=10)
    ax.set(xticks=range(4), xticklabels=['CPU 4', 'CPU 8', 'CPU 20', 'GPU + CPU 8'],
           ylabel='Complete-job wall time (s)', ylim=(0, 102))
    ax.grid(axis='y', alpha=.2)
    ax.text(.99, .97, '200 steps · 40³ · 20 orbitals\n3 trials/configuration; dots = trials', transform=ax.transAxes, ha='right', va='top', fontsize=9)
    save(fig, 'benchmark')
    profile = final('profile')
    keys = ['predictor', 'midpoint_fields_and_upload', 'corrector', 'endpoint_fields_and_diagnostics']
    labels = ['Predictor', 'Midpoint fields', 'Corrector', 'Endpoint fields + diagnostics']
    stages = {r['backend']: {s['stage']: s['wall_seconds'] for s in r['stages']} for r in profile['runs']}
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(7.1, 3.7), constrained_layout=True)
    for offset, backend, color in [(-.18, 'cpu', BLUE), (.18, 'gpu', ORANGE)]:
        values = [stages[backend][k] for k in keys]
        ax.barh(y + offset, values, height=.32, label=backend.upper(), color=color)
        for pos, v in zip(y + offset, values): ax.text(v + .14, pos, f'{v:.3f}', va='center', fontsize=9)
    ax.set(yticks=y, yticklabels=labels, xlabel='Exclusive accumulated wall time (s)', xlim=(0, 15.8))
    ax.invert_yaxis(); ax.legend(frameon=False); ax.grid(axis='x', alpha=.2)
    save(fig, 'stages')
    rows = [['Exclusive stage', 'CPU (s)', 'GPU (s)']]
    for s in profile['runs'][0]['stages']:
        if not s['nested_in_other_stages']:
            rows.append([s['stage'].replace('_', r'\_'), f"{s['wall_seconds']:.6f}", f"{stages['gpu'][s['stage']]:.6f}"])
    tex_table('stages', 'lrr', rows)
    p = final('preflight')
    rows = [['Grid / orbitals', 'Arrays + workspace (GiB)', 'Decision']]
    memory = []
    for c in p['cases']:
        grid = c['grid'][0]; value = c['required_bytes']/2**30
        memory.append(value)
        rows.append([r'$' + str(grid) + r'^3$ / ' + str(c['states']), f'{value:.6f}', c['status']])
    tex_table('memory', 'lrl', rows)
    fig, ax = plt.subplots(figsize=(7.1, 3.9), constrained_layout=True)
    ax.bar(range(4), memory, color=[BLUE, BLUE, BLUE, ORANGE], width=.6)
    for i, v in enumerate(memory): ax.text(i, v+.18, f'{v:.3f}', ha='center')
    free = p['cases'][0]['free_bytes']/2**30
    ax.axhline(free*.8, ls='--', color='black', label=f'80% of queried free VRAM ({free*.8:.2f} GiB)')
    ax.set(xticks=range(4), xticklabels=['24³ / 20', '40³ / 20', '48³ / 208', '64³ / 208'],
           ylabel='Required device memory (GiB)', ylim=(0, 17))
    ax.legend(frameon=False, loc='upper left', fontsize=9); ax.grid(axis='y', alpha=.2)
    save(fig, 'memory')
    ind = final('independent-nuclei')
    rows = [['Case / mesh', 'Stage', 'CPU (s)', 'GPU (s)', 'Ratio']]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.8), constrained_layout=True)
    for ax, c in zip(axes, ind['cases']):
        cpu = [c['static_seconds']['cpu'], c['dynamic_seconds']['cpu']]
        gpu = [c['static_seconds']['gpu'], c['dynamic_seconds']['gpu']]
        case = r'$^{16}$O / 40³' if c['states']==16 else r'$^{20}$Ne VDI / 56³'
        ax.bar(np.arange(2)-.17, cpu, width=.32, color=BLUE, label='CPU')
        ax.bar(np.arange(2)+.17, gpu, width=.32, color=ORANGE, label='GPU')
        ax.set(xticks=[0,1], xticklabels=['Static', 'TDHF 10 fm/c'], ylabel='Complete-job time (s)', title=case)
        ax.grid(axis='y', alpha=.2); ax.legend(frameon=False, fontsize=9)
        for i,(a,z) in enumerate(zip(cpu,gpu)): ax.text(i, max(a,z)*1.06, f'{a/z:.3f}×', ha='center')
        ax.set_ylim(0,max(cpu)*1.25)
        label = r'$^{16}$O / $40^3$' if c['states']==16 else r'$^{20}$Ne VDI / $56^3$'
        for stage, a, z in zip(['Static', 'TDHF'],cpu,gpu): rows.append([label,stage,f'{a:.6f}',f'{z:.6f}',f'{a/z:.4f}'])
    tex_table('independent', 'llrrr', rows)
    save(fig, 'independent')
    # The earlier matched-box CPU refinement study is distinct from the two-state endpoint probe.
    grid_rows = []
    for n, f in [(24,'static-convergence-24'),(32,'static-convergence-32'),(40,'static-convergence-40-polished')]:
        d=load(REPO/'CUDA/results'/(f+'.json'))
        grid_rows.append((n,d))
    # Values below are the independently reported occupation-weighted means in STATIC_CONVERGENCE.md.
    fig, ax=plt.subplots(figsize=(7.1,3.7),constrained_layout=True)
    ax.semilogy([24,32,40],[1.223e-4,2.157e-6,1.598e-8],'o-',color=BLUE,label='Fresh direct residual')
    ax.semilogy([24,32,40],[7.559e-5,2.629e-6,1.101e-7],'s--',color=ORANGE,label=r'Fresh $H^2$ fluctuation')
    ax.axhline(1e-6,color='black',ls=':',label='Original 1e−6 MeV threshold')
    ax.set(xticks=[24,32,40],xticklabels=['24³ / 1 fm','32³ / 0.75 fm','40³ / 0.6 fm'],ylabel='Weighted mean per state (MeV)')
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.2,which='both')
    save(fig,'static-refinement')
    ori=final('orientation')
    fig, ax=plt.subplots(figsize=(7.1,3.7),constrained_layout=True)
    for off,key,color in [(-.17,'original',BLUE),(.17,'rotated',ORANGE)]:
        ax.bar(np.arange(3)+off,ori[key]['coordinate_mean_square_fm2'],width=.32,color=color,label=key.capitalize())
    ax.set(xticks=[0,1,2],xticklabels=['x','y','z'],ylabel=r'Mean square coordinate (fm$^2$)',ylim=(0,5.4))
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.2)
    save(fig,'orientation')
    q=load(QUAD/'quadrupole-comparison.json')
    rows=[['Width (MeV)', 'GPU / current CPU', 'GPU / archived CPU', 'Archived reanalysis']]
    for name,width in [('gamma_1p0','1.0'),('gamma_0p5','0.5'),('gamma_0p2','0.2')]:
        c=q['all_smoothing_comparisons'][name]
        rows.append([width]+[sci(c[k]['relative_l2']) for k in ['gpu_vs_current_cpu','gpu_vs_saved_cpu','archived_analysis_reproduction']])
    tex_table('smoothing','lrrr',rows)
    sys.path.insert(0,str(REPO/'CUDA'))
    from compare_quadrupole import prefix, transform, plots
    archived=REPO/'projects/icnpa2026_20Ne_Kresolved_E2/response/prefix_runs'
    raw=Path(q['run_directory'])
    paths={'CPU':raw/'cpu/quadrupoles.res','GPU':raw/'gpu/quadrupoles.res','Saved CPU':archived/'K0/quadrupoles.res'}
    reference=archived/'reference_K0/quadrupoles.res'
    signals={k:prefix(v,6000) for k,v in paths.items()}
    oldref=prefix(reference,6000)
    spectra={k:transform(v,oldref) for k,v in signals.items()}
    smooth={g:{k:transform(v,oldref,g) for k,v in signals.items()} for g in (1.,.5,.2)}
    # Exact reanalysis must reproduce saved numerical comparisons before figure export.
    mask=(spectra['CPU'].energy_mev>=.5)&(spectra['CPU'].energy_mev<=35)
    a=spectra['GPU'].signed_strength[mask]; z=spectra['CPU'].signed_strength[mask]
    err=float(np.linalg.norm(a-z)/np.linalg.norm(z))
    assert abs(err-q['spectrum_comparisons']['gpu_vs_current_cpu']['relative_l2'])<1e-15
    plots(ASSETS,signals,spectra,smooth)
    response_inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths.values(),reference]}
    # Keep exact report and code identities beside all generated assets.
    for name in ['validation','extended-validation','mode-controls','static-controls','static-diagonalization-off',
                 'static-fallback-before-fix','energy-fix-validation','initial-flow-cpu-control','sanitizer',
                 'response-endpoint','response-reference-grid-audit','independent-nuclei-coarse',
                 'independent-nuclei-40','paired48','paired56-dt01-failure','paired56','own-gpu-16o',
                 'own-gpu-paired56','clone-validation','preservation-check','onboarding-validation','rotation']:
        final(name)
    srcpaths=['CUDA/sky_gpu.cu','CUDA/reductions.cuh','CUDA/gpu_runtime.f90','Code/gpu_runtime.f90',
              'Code/dynamic.f90','Code/static.f90','Code/moment.f90','Code/meanfield.f90','CUDA/build.py',
              'Utils/Strength_Calculation/sky3d_response/analysis.py','docs/STATIC_CONVERGENCE.md']
    source_sha={s:hashlib.sha256((REPO/s).read_bytes()).hexdigest() for s in srcpaths}
    manifest={'schema_version':1,'source_snapshot':'9576125da0372c5cb3092032fafad32a50339bd2',
        'report_generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evidence_sha256':inputs,'source_sha256':source_sha,'raw_response_sha256':response_inputs,
        'response_reanalysis_relative_l2':err,
        'note':'Saved evidence reanalysis only; no new simulation or timing. Absolute raw paths are local provenance, not clone prerequisites.'}
    (HERE/'evidence_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=[]
    for path,digest in inputs.items():
        lines.extend([r'\noindent\path{'+path+r'}\par',r'{\footnotesize\ttfamily '+digest+r'}\par\smallskip'])
    (TABLES/'evidence_hashes.tex').write_text('\n'.join(lines)+'\n')
    lines=[]
    for label,digest in [('Fine 20Ne initial checkpoint',b['state_sha256']),('Archived K=0 initial checkpoint',q['state_sha256']),
                         ('Final benchmark GPU executable',b['build']['executable_sha256']['gpu']),
                         ('Final benchmark CUDA library',b['build']['library_sha256']),
                         ('Response GPU executable',q['build']['executable_sha256']['gpu']),
                         ('Response CUDA library',q['build']['library_sha256'])]:
        lines.extend([r'\noindent '+label+r'\par',r'{\footnotesize\ttfamily '+digest+r'}\par\smallskip'])
    (TABLES/'key_hashes.tex').write_text('\n'.join(lines)+'\n')
    print('Generated report assets and tables; response reanalysis relative L2:',err)

if __name__=='__main__':
    main()
