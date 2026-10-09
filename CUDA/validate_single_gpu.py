#!/usr/bin/env python3
"""Single-GPU static and mode coverage; all raw output stays in Linux cache."""
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
from benchmark import REPO, checkpoint, densities, difference, input_text, compare
from validate_extended import sha256, verify_build, exact_steps
from static_convergence import build_probe, analyse


def execute(exe, path, text, backend, filename, settings=None, restart=None):
    path.mkdir()
    (path/'for005').write_text(text)
    if restart:
        shutil.copy2(restart, path/filename)
    env = dict(os.environ, SKY3D_BACKEND=backend, OMP_NUM_THREADS='8',
               OPENBLAS_NUM_THREADS='1', OMP_PROC_BIND='close', OMP_PLACES='cores')
    env.pop('SKY3D_GPU_VALIDATE', None)
    env.update(settings or {})
    start = time.perf_counter()
    with (path/'stdout.log').open('w') as log:
        subprocess.run([str(exe)], cwd=path, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    seconds = time.perf_counter()-start
    state = checkpoint(path/filename)  # A Fortran STOP can return zero: require output.
    row = dict(directory=str(path), seconds=seconds, input_sha256=sha256(path/'for005'),
               checkpoint_sha256=sha256(path/filename), step=state['step'],
               particles=state['particles'], settings=settings or {})
    print(path.name, f'{seconds:.3f}s', flush=True)
    return row, state


def static_input(nucleons, force, pairing='NONE', steps=120, converged=False, mesh=32, spacing=0.8, serr=1e-6):
    extra = 'npsi=18,18,' if pairing != 'NONE' else ''
    radius = '2.6, radiny=2.6, radinz=2.6' if nucleons == 8 else '2.4, radiny=2.4, radinz=4.0'
    stopping=f'{serr:.16e}'.replace('e','D') if converged else '1.0D-20'
    return f"""&files wffile='static.tdhf' /
&force name='{force}', pairing='{pairing}' /
&main imode=1, nof=0, tfft=T, mprint=10, mplot={steps}, mrest={steps},
 writeselect='rtcsouw', write_isospin=T /
&grid nx={mesh}, ny={mesh}, nz={mesh}, dx={spacing}, dy={spacing}, dz={spacing}, periodic=F /
&static nneut={nucleons}, nprot={nucleons}, {extra}
 radinx={radius}, tdiag=T, tlarge=F,
 x0dmp=0.4, e0dmp=100.0, maxiter={steps}, serr={stopping} /
"""


def density_from_state(state):
    basis=state['psi'].reshape(state['states'], 2, -1)
    return np.sum(state['attributes'][0,:,None]*np.sum(np.abs(basis)**2,axis=1),axis=0)


def static_comparison(actual, reference):
    errors = {name:difference(value,densities(reference)[name])
              for name,value in densities(actual).items()}
    for name,error in errors.items():
        limit=1e-9+1e-10*np.max(np.abs(densities(reference)[name]))
        if error['max_abs'] > limit:
            raise AssertionError((name,error,limit))
    return errors


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--state', type=Path, help='Validated 20Ne seed required for controls; preparation generates its own states')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('controls','prepare'), default='controls')
    parser.add_argument('--maxiter', type=int, default=4000)
    parser.add_argument('--prepare-mesh',type=int,default=32)
    parser.add_argument('--prepare-spacing',type=float,default=0.8)
    parser.add_argument('--prepare-case',choices=('all','16o-sly5','20ne-sly4-vdi'),default='all')
    parser.add_argument('--prepare-serr',type=float,default=1e-6,help='Explicit original or stricter static criterion, at most 1e-6 MeV')
    parser.add_argument('--dynamic-dt',type=float,default=0.1)
    args=parser.parse_args()
    if args.mode=='controls' and args.state is None:
        parser.error('--state is required for dynamic mode controls')
    if args.prepare_mesh<8 or args.prepare_mesh%2 or not np.isfinite(args.prepare_spacing) or args.prepare_spacing<=0 or args.maxiter<1:
        parser.error('Positive iterations/spacing and an even preparation mesh >=8 required')
    if not np.isfinite(args.prepare_serr) or not 0<args.prepare_serr<=1e-6:
        parser.error('Preparation serr must be positive and cannot exceed the original 1e-6 MeV limit')
    dynamic_steps=exact_steps(10,args.dynamic_dt)
    build,exes=verify_build(args.build_dir.resolve())
    root=Path(tempfile.mkdtemp(prefix='sky3d-single-coverage-',dir=Path.home()/'.cache'))
    report=dict(run_directory=str(root),build=build,runner_sha256=sha256(Path(__file__)),
                mode=args.mode,cases=[],passed=False)
    if args.mode=='prepare':
        report['preparation_parameters']=dict(mesh=args.prepare_mesh,spacing_fm=args.prepare_spacing,
            maxiter=args.maxiter,serr_mev=args.prepare_serr,case=args.prepare_case)
        report['dynamic_parameters']=dict(dt_fm_c=args.dynamic_dt,steps=dynamic_steps,endpoint_fm_c=10)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        if args.mode=='controls':
            for label,n,force,pairing in [('16o-sly5',8,'Sly5','NONE'),
                                         ('20ne-sly4-vdi',10,'Sly4','VDI')]:
                row=dict(case=label,kind='fixed-iteration implementation control',runs={})
                report['cases'].append(row)
                for backend in ('cpu','gpu'):
                    path=root/f'{label}-{backend}'
                    timing,state=execute(exes[backend],path,static_input(n,force,pairing),backend,'static.tdhf')
                    if state['step']!=120:
                        raise AssertionError('Static iteration endpoint missing')
                    if max(abs(p-n) for p in state['particles'])>1e-6:
                        raise AssertionError('Static particle conservation')
                    timing['occupation_range']=[float(state['attributes'][0].min()),float(state['attributes'][0].max())]
                    timing['fractional_orbitals']=int(np.sum((state['attributes'][0]>1e-8)&(state['attributes'][0]<1-1e-8)))
                    row['runs'][backend]=timing
                    if backend=='gpu':
                        timing['fields']=static_comparison(path/'000120.tdd',root/f'{label}-cpu/000120.tdd')
                    save()
                if pairing!='NONE' and row['runs']['gpu']['fractional_orbitals']==0:
                    raise AssertionError('Pairing control did not exercise fractional occupations')
                row['passed']=True
                save()
            shutil.copy2(args.state,root/'initial.tdhf')
            state=checkpoint(args.state)
            modes=[('periodic',{},'periodic'),('no-coulomb',{},'no-coulomb'),
                   ('graphs-off',{'SKY3D_GPU_GRAPHS':'0'},None),
                   ('cpu-fields',{'SKY3D_GPU_FIELDS':'0'},None),
                   ('cpu-diagnostics',{'SKY3D_GPU_DIAGNOSTICS':'0'},None),
                   ('no-residency',{'SKY3D_GPU_RESIDENT':'0'},None),
                   ('positive-M1',{},1),('positive-M2',{},2),('negative-M2',{},-2)]
            for label,settings,change in modes:
                text=input_text(20,state['grid'],validate=True,spacing=state['spacing'],dt=0.1)
                if change=='periodic':
                    text=text.replace('periodic=F','periodic=T')
                elif change=='no-coulomb':
                    text=text.replace('imode=2,','imode=2, tcoul=F,')
                elif isinstance(change,int):
                    text=re.sub(r'\bM_val\s*=\s*0',f'M_val={change}',text,flags=re.I)
                row=dict(case=label,kind='dynamic mode control',runs={})
                report['cases'].append(row)
                for backend in ('cpu','gpu'):
                    path=root/f'{label}-{backend}'
                    timing,final=execute(exes[backend],path,text,backend,'k0_restart.tdhf',settings)
                    if final['step']!=20 or abs(final['time']-2)>1e-9:
                        raise AssertionError('Dynamic endpoint missing')
                    if backend=='gpu':
                        timing['comparison']=compare(path,root/f'{label}-cpu',1e-5)
                    row['runs'][backend]=timing
                    save()
                row['passed']=True
                save()
        else:
            # Independently converge each nucleus/force before the dynamic check.
            probe_exe,probe_build=build_probe(root)
            report["fresh_field_probe_build"]=probe_build
            save()
            for label,n,force,pairing in [('16o-sly5',8,'Sly5','NONE'),
                                         ('20ne-sly4-vdi',10,'Sly4','VDI')]:
                if args.prepare_case!='all' and label!=args.prepare_case:
                    continue
                row=dict(case=label,kind='converged independent CPU preparation plus CPU/GPU TDHF',runs={})
                report['cases'].append(row)
                path=root/f'{label}-static'
                preparation_input=static_input(n,force,pairing,args.maxiter,True,args.prepare_mesh,args.prepare_spacing,args.prepare_serr)
                timing,state=execute(exes['cpu'],path,preparation_input,'cpu','static.tdhf')
                row['runs']['static']=timing
                save()
                log=(path/'stdout.log').read_text()
                if 'Static convergence criterion met at iteration' not in log:
                    raise AssertionError(f'{label}: requested serr not reached; raw calculation preserved at {path}')
                match=re.search(r'Final pre-gradient fluctuations \(MeV\): h\*\*2=\s*(\S+)',log)
                timing['pre_gradient_fluctuation_mev']=float(match.group(1).rstrip(','))
                if timing['pre_gradient_fluctuation_mev']>=args.prepare_serr:
                    raise AssertionError('Static convergence message and recorded criterion disagree')
                timing['fractional_orbitals']=int(np.sum((state['attributes'][0]>1e-8)&(state['attributes'][0]<1-1e-8)))
                if pairing!='NONE' and timing['fractional_orbitals']==0:
                    raise AssertionError('Converged paired fixture must exercise fractional occupations')
                save()
                gpu_path=root/f'{label}-static-gpu'
                gpu_timing,gpu_state=execute(exes['gpu'],gpu_path,
                    preparation_input,'gpu','static.tdhf')
                row['runs']['static-gpu']=gpu_timing
                save()
                gpu_log=(gpu_path/'stdout.log').read_text()
                if 'Static convergence criterion met at iteration' not in gpu_log:
                    raise AssertionError(f'{label}: GPU static serr not reached')
                gm=re.search(r'Final pre-gradient fluctuations \(MeV\): h\*\*2=\s*(\S+)',gpu_log)
                gpu_timing['pre_gradient_fluctuation_mev']=float(gm.group(1).rstrip(','))
                if gpu_timing['pre_gradient_fluctuation_mev']>=args.prepare_serr:
                    raise AssertionError('GPU static recorded convergence criterion')
                gpu_timing['density_from_checkpoint']=difference(density_from_state(gpu_state),density_from_state(state))
                if gpu_timing['density_from_checkpoint']['max_abs']>1e-6:
                    raise AssertionError('Independently converged static density mismatch')
                volume=float(np.prod(state['spacing']))
                split=state['states']//2
                singular=[]
                for sl in (slice(0,split),slice(split,None)):
                    overlap=volume*state['psi'][sl].conj()@gpu_state['psi'][sl].T
                    singular.append(float(np.linalg.svd(overlap,compute_uv=False).min()))
                gpu_timing['per_species_subspace_min_singular_values']=singular
                if min(singular)<1-1e-6:
                    raise AssertionError('Converged static subspace mismatch')
                row['runs']['static-gpu']=gpu_timing
                row['static_time_to_solution_speedup_one_pair']=timing['seconds']/gpu_timing['seconds']
                row['fresh_field_probes']={}
                for tag,seed_path in [('cpu',path/'static.tdhf'),('gpu',gpu_path/'static.tdhf')]:
                    probe_path=root/f'{label}-fresh-{tag}'
                    probe_path.mkdir()
                    probe_text=preparation_input.replace('nof=0','nof=1')
                    probe_text+=f"&fragments filename='{seed_path}', fix_boost=T /\n"
                    (probe_path/'for005').write_text(probe_text)
                    with (probe_path/'stdout.log').open('w') as log:
                        subprocess.run([str(probe_exe)],cwd=probe_path,stdout=log,stderr=subprocess.STDOUT,check=True,
                                       env=dict(os.environ,SKY3D_BACKEND='cpu',OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1'))
                    metrics=analyse(probe_path/'probe.bin',state['spacing'],state['states']//2)
                    row['fresh_field_probes'][tag]=metrics
                    save()
                    if metrics['weighted_direct_residual_per_state_mev']>1e-6:
                        raise AssertionError('Fresh-field residual exceeds explicit 1e-6 MeV additional-state gate')
                    for species in metrics['species']:
                        if species['orthonormality_max_abs']>1e-10 or species['hamiltonian_antihermiticity_max_abs_mev']>1e-8:
                            raise AssertionError('Fresh-field Gram or Hermiticity gate')
                energies=[row['fresh_field_probes'][tag]['fresh_field_observables']['integrated_energy_mev'] for tag in ('cpu','gpu')]
                row['fresh_field_integrated_energy_difference_mev']=abs(energies[0]-energies[1])
                if abs(energies[0]-energies[1])>1e-6:
                    raise AssertionError('Converged static fresh-field energy mismatch')
                row['fresh_field_residual_limit_mev']=1e-6
                row['static_stopping_serr_mev']=args.prepare_serr
                save()
                shutil.copy2(path/'static.tdhf',root/'initial.tdhf')
                text=input_text(dynamic_steps,state['grid'],validate=True,spacing=state['spacing'],dt=args.dynamic_dt)
                text=text.replace("name='Sly5', pairing='NONE'",f"name='{force}', pairing='{pairing}'")
                for backend in ('cpu','gpu'):
                    target=root/f'{label}-td-{backend}'
                    result,final=execute(exes[backend],target,text,backend,'k0_restart.tdhf')
                    if final['step']!=dynamic_steps or abs(final['time']-10)>1e-8:
                        raise AssertionError('Dynamic endpoint missing')
                    if max(abs(p-n) for p in final['particles'])>1e-6:
                        raise AssertionError('Dynamic particle conservation')
                    result['occupation_max_drift']=float(np.max(np.abs(final['attributes'][0]-state['attributes'][0])))
                    if result['occupation_max_drift']>1e-12:
                        raise AssertionError('Frozen occupation TDHF changed occupations')
                    if backend=='gpu':
                        result['comparison']=compare(target,root/f'{label}-td-cpu',1e-5)
                    row['runs'][backend]=result
                    save()
                row['passed']=True
                save()
        report['passed']=True
        save()
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        save()
        raise
    print('Single-GPU coverage passed:',args.output,flush=True)


if __name__=='__main__':
    main()
