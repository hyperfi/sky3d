#!/usr/bin/env python3
"""Reuse independently validated static states for a new explicit TDHF timestep."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from benchmark import checkpoint,input_text,compare
from validate_single_gpu import execute
from validate_extended import sha256,verify_build,exact_steps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--case',choices=('16o-sly5','20ne-sly4-vdi'),required=True)
    parser.add_argument('--dt',type=float,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    steps=exact_steps(10,args.dt)
    build,exes=verify_build(args.build_dir.resolve())
    source=json.loads(args.preparation.read_text())
    if source['build']['executable_sha256']!=build['executable_sha256']:
        raise ValueError('Require the exact frozen static build')
    case=copy.deepcopy(next(x for x in source['cases'] if x['case']==args.case))
    for tag in ('cpu','gpu'):
        static=case['runs']['static' if tag=='cpu' else 'static-gpu']
        if static['pre_gradient_fluctuation_mev']>=case['static_stopping_serr_mev']:
            raise ValueError('Original static stopping gate missing')
        probe=case['fresh_field_probes'][tag]
        if probe['weighted_direct_residual_per_state_mev']>1e-6 or any(
            x['orthonormality_max_abs']>1e-10 or x['hamiltonian_antihermiticity_max_abs_mev']>1e-8 for x in probe['species']):
            raise ValueError('Cannot reuse a static state that failed fresh-field validation')
    root=Path(tempfile.mkdtemp(prefix='sky3d-prepared-resume-',dir=Path.home()/'.cache'))
    seed=Path(case['runs']['static']['directory'])/'static.tdhf'
    if sha256(seed)!=case['runs']['static']['checkpoint_sha256']:
        raise ValueError('Original static seed changed')
    state=checkpoint(seed)
    split=state['states']//2
    volume=float(np.prod(state['spacing']))
    initial_gram=[volume*p.conj()@p.T for p in (state['psi'][:split],state['psi'][split:])]
    shutil.copy2(seed,root/'initial.tdhf')
    text=input_text(steps,state['grid'],validate=True,spacing=state['spacing'],dt=args.dt)
    force,pairing=('Sly5','NONE') if args.case=='16o-sly5' else ('Sly4','VDI')
    text=text.replace("name='Sly5', pairing='NONE'",f"name='{force}', pairing='{pairing}'")
    case['passed']=False
    report=dict(run_directory=str(root),build=build,runner_sha256=sha256(Path(__file__)),
        source_preparation_sha256=sha256(args.preparation),source_preparation=str(args.preparation.resolve()),
        preparation_parameters=source['preparation_parameters'],
        dynamic_parameters=dict(dt_fm_c=args.dt,steps=steps,endpoint_fm_c=10),
        cases=[case],passed=False,
        note='Static measurements are preserved from the referenced preparation; only the explicit dynamic timestep is rerun.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        for backend in ('cpu','gpu'):
            target=root/f'{args.case}-td-{backend}'
            row,final=execute(exes[backend],target,text,backend,'k0_restart.tdhf')
            case['runs'][backend]=row
            save()
            if final['step']!=steps or abs(final['time']-10)>1e-8:
                raise AssertionError('Dynamic endpoint missing')
            if max(abs(a-b) for a,b in zip(final['particles'],state['particles']))>1e-6:
                raise AssertionError('Dynamic particle conservation')
            row['occupation_max_drift']=float(np.max(np.abs(final['attributes'][0]-state['attributes'][0])))
            if row['occupation_max_drift']>1e-12:
                raise AssertionError('Frozen occupations changed')
            matrices=[volume*p.conj()@p.T for p in (final['psi'][:split],final['psi'][split:])]
            row['gram_max_abs_error']=float(max(np.max(np.abs(g-np.eye(len(g)))) for g in matrices))
            row['gram_max_abs_drift']=float(max(np.max(np.abs(g-g0)) for g,g0 in zip(matrices,initial_gram)))
            if max(row['gram_max_abs_error'],row['gram_max_abs_drift'])>1e-6:
                raise AssertionError('Occupied Gram gate')
            if backend=='gpu':
                row['comparison']=compare(target,root/f'{args.case}-td-cpu',1e-5)
            save()
        case['passed']=True
        report['passed']=True
        save()
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        save()
        raise
    print('Prepared-state dynamics passed:',args.output,flush=True)


if __name__=='__main__':
    main()
