#!/usr/bin/env python3
"""Check TDHF from the independently GPU-prepared fixture, using physical outputs."""
import argparse
import json
import os
from pathlib import Path
import re
import tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
from benchmark import checkpoint,densities,difference,input_text,check_observable
from validate_single_gpu import execute
from validate_extended import sha256,verify_build


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir',type=Path,required=True)
    parser.add_argument('--preparation',type=Path,required=True)
    parser.add_argument('--case',choices=('16o-sly5','20ne-sly4-vdi'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build,exes=verify_build(args.build_dir.resolve())
    preparation=json.loads(args.preparation.read_text())
    if preparation['build']['executable_sha256']!=build['executable_sha256']:
        raise ValueError('Preparation must use this exact frozen build')
    case=next(x for x in preparation['cases'] if x['case']==args.case)
    if not case.get('passed'):
        raise ValueError('Require a passed independent static plus matched TDHF case')
    seed=Path(case['runs']['static-gpu']['directory'])/'static.tdhf'
    expected_hash=case['runs']['static-gpu']['checkpoint_sha256']
    if sha256(seed)!=expected_hash:
        raise ValueError('GPU-prepared seed changed')
    initial=checkpoint(seed)
    reference=Path(case['runs']['cpu']['directory'])
    text=(reference/'for005').read_text()
    steps=int(re.search(r'\bnt\s*=\s*(\d+)',text,re.I).group(1))
    dt=float(re.search(r'\bdt\s*=\s*([\d.eEdD+-]+)',text,re.I).group(1).replace('D','E').replace('d','e'))
    text=text.replace("'../initial.tdhf'","'"+str(seed).replace("'","''")+"'")
    root=Path(tempfile.mkdtemp(prefix='sky3d-own-preparation-',dir=Path.home()/'.cache'))
    report=dict(case=args.case,build=build,preparation_sha256=sha256(args.preparation),
        runner_sha256=sha256(Path(__file__)),initial_state_sha256=expected_hash,
        run_directory=str(root),passed=False,
        note='Compare fields and printed observables: independently prepared orbitals may differ by phases or degenerate-space rotations.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    save()
    try:
        row,final=execute(exes['gpu'],root/'gpu',text,'gpu','k0_restart.tdhf')
        report['run']=row
        if final['step']!=steps or abs(final['time']-steps*dt)>1e-8:
            raise AssertionError('Incomplete own-preparation trajectory')
        if sha256(seed)!=expected_hash:
            raise AssertionError('TDHF modified the supplied GPU preparation')
        row['occupation_max_drift']=float(np.max(np.abs(final['attributes'][0]-initial['attributes'][0])))
        if row['occupation_max_drift']>1e-12 or max(abs(a-b) for a,b in zip(final['particles'],initial['particles']))>1e-6:
            raise AssertionError('Frozen occupations or particle conservation')
        split=initial['states']//2
        matrices=[float(np.prod(final['spacing']))*p.conj()@p.T for p in (final['psi'][:split],final['psi'][split:])]
        row['gram_max_abs_error']=float(max(np.max(np.abs(g-np.eye(len(g)))) for g in matrices))
        if row['gram_max_abs_error']>1e-6:
            raise AssertionError('Own-preparation occupied-state Gram gate')
        a=densities(root/'gpu'/f'{steps:06d}.tdd');b=densities(reference/f'{steps:06d}.tdd')
        row['fields']={key:difference(a[key],b[key]) for key in b}
        save()
        for key,error in row['fields'].items():
            if error['max_abs']>1e-9+1e-10*np.max(np.abs(b[key]),initial=0):
                raise AssertionError(('Own-preparation field',key,error))
        row['printed_observables']={}
        for name in ('energies.res','quadrupoles.res','monopoles.res'):
            actual=np.loadtxt(root/'gpu'/name,ndmin=2);target=np.loadtxt(reference/name,ndmin=2)
            row['printed_observables'][name]=difference(actual,target)
            save()
            check_observable(name,actual,target,1e-5)
        report['passed']=True
        save()
    except Exception as error:
        report['error']=f'{type(error).__name__}: {error}'
        save()
        raise
    print('Own GPU-prepared TDHF check passed:',args.output,flush=True)


if __name__=='__main__':
    main()
