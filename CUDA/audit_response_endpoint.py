#!/usr/bin/env python3
"""Audit saved response endpoints without rerunning or modifying a trajectory."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from benchmark import checkpoint
from validate_extended import gram, sha256


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    comparison=json.loads(args.comparison.read_text())
    root=Path(comparison['run_directory'])
    initial_path=root/'initial.tdhf'
    if not comparison['passed'] or sha256(initial_path)!=comparison['state_sha256']:
        raise ValueError('Require a passed comparison and its original matched checkpoint')
    state=checkpoint(initial_path)
    if state['states']!=sum(state['nucleons']):
        raise ValueError('Occupied-only response audit; extra orbitals require species-aware overlaps')
    physics=comparison['physics']
    report=dict(comparison_sha256=sha256(args.comparison),runner_sha256=sha256(Path(__file__)),
        initial_state_sha256=sha256(initial_path),physics=physics,runs={},passed=False,
        cpu_gpu_equivalence_passed=comparison['passed'],
        limits=dict(gram_max_abs_error_and_drift=1e-6,
                    particle_atol=comparison['tolerances']['checkpoint_particle_atol'],
                    physical_energy_drift='recorded at native printed precision; no production accuracy pass'))
    matrices_by_backend={}
    initial_gram=gram(state)
    for backend in ('cpu','gpu'):
        path=root/backend
        final=checkpoint(path/'k0_restart.tdhf')
        matrices=gram(final)
        matrices_by_backend[backend]=matrices
        energy=np.loadtxt(path/'energies.res',ndmin=2)
        row=dict(gram_max_abs_error=float(max(np.max(np.abs(g-np.eye(len(g)))) for g in matrices)),
            gram_max_abs_drift=float(max(np.max(np.abs(g-g0)) for g,g0 in zip(matrices,initial_gram))),
            integrated_energy_initial_mev=float(energy[0,4]),integrated_energy_final_mev=float(energy[-1,4]),
            integrated_energy_max_drift_mev=float(np.max(np.abs(energy[:,4]-energy[0,4]))),
            final_particles=final['particles'],dt_fm_c=physics['dt_fm_c'],taylor_order=4,
            endpoint_complete=final['step']==physics['steps'] and abs(final['time']-physics['endpoint_fm_c'])<=1e-8,
            finite=all(np.isfinite(g).all() for g in matrices) and bool(np.isfinite(energy).all()))
        drift=max(abs(a-b) for a,b in zip(final['particles'],state['particles']))
        row.update(particle_max_drift=drift,checkpoint_sha256=sha256(path/'k0_restart.tdhf'))
        row['passed']=row['endpoint_complete'] and row['finite'] and drift<=report['limits']['particle_atol'] and max(
            row['gram_max_abs_error'],row['gram_max_abs_drift'])<=report['limits']['gram_max_abs_error_and_drift']
        report['runs'][backend]=row
    report['cpu_gpu_gram_max_abs_difference']=float(max(np.max(np.abs(a-b)) for a,b in zip(
        matrices_by_backend['cpu'],matrices_by_backend['gpu'])))
    report['passed']=all(row['passed'] for row in report['runs'].values())
    report['note']='Physical gates are separate from CPU/GPU equivalence. Failed gates remain failures; both backend endpoints and original limits are recorded.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved response endpoint audit:',args.output,'physical gates passed:',report['passed'],flush=True)
    return 0 if report['passed'] else 2


if __name__=='__main__':
    sys.exit(main())
