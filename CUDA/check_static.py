#!/usr/bin/env python3
"""Compare static densities across thread counts after the basis-read fix."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from benchmark import REPO, checkpoint, densities, difference
from validate_extended import verify_build, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=REPO / 'CUDA/build')
    parser.add_argument('--output', type=Path, default=REPO / 'CUDA/results/static.json')
    parser.add_argument('--diagonalization',choices=('on','off'),default='on')
    args = parser.parse_args()
    build,executables=verify_build(args.build_dir.resolve())
    root = Path(tempfile.mkdtemp(prefix='sky3d-static-check-', dir=Path.home() / '.cache'))
    text = (REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/static_20ne_sly5.in').read_text()
    text = text.replace('maxiter=3000', 'maxiter=120').replace('mrest=200', 'mrest=120').replace('mplot=100000', 'mplot=120')
    text = text.replace("writeselect='r'", "writeselect='rtcsouw', write_isospin=T")
    if args.diagonalization=='off':
        if text.count('tdiag=T')!=1:
            raise ValueError('Static diagonalization input anchor changed')
        text=text.replace('tdiag=T','tdiag=F')
    rows = []
    reference = None
    reference_state = None
    configurations=[(name,threads,prints,{}) for name,threads,prints in
        [('reference',1,10),('reference',8,10),('reference',8,10),('cpu',8,10),
         ('cpu',8,0),('gpu',8,10),('gpu',8,0)]]
    configurations += [('gpu',8,10,{setting:'0'}) for setting in
        ('SKY3D_GPU_FIELDS','SKY3D_GPU_DIAGNOSTICS','SKY3D_GPU_GRAPHS')]
    def save(passed=False):
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(run_directory=str(root),steps=120,passed=passed,
            cases=rows,build=build,runner_sha256=sha256(Path(__file__)),diagonalization=args.diagonalization,
            note='Static fields and printed geometry; not individual orbital phases or convergence proof'),indent=2)+'\n')
    save()
    geometry_reference=None
    for name, threads, prints, settings in configurations:
        path = root / f'{len(rows)}-{name}-{threads}'
        path.mkdir()
        (path / 'for005').write_text(text.replace('mprint=10', f'mprint={prints}'))
        exe = executables[name]
        with (path / 'stdout.log').open('w') as log:
            subprocess.run([str(exe)], cwd=path, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, SKY3D_BACKEND='gpu' if name == 'gpu' else 'cpu', OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1',**settings))
        state = checkpoint(path / '20ne_sly5.tdhf')
        assert state['step'] == 120
        if prints:
            current = densities(path / '000120.tdd')
        else:
            # Static density output is nested under printing. Compare rho
            # directly from the final checkpoint when that output is disabled.
            psi = state['psi'].reshape(state['states'], 2, -1)
            current = {'Rho': np.sum(state['attributes'][0, :, None] * np.sum(np.abs(psi)**2, axis=1), axis=0)}
        if reference is None:
            reference = current
            reference_state = state
        if prints:
            target = reference
        else:
            # The static .tdd rho is relaxed/mixed, unlike checkpoint rho.
            basis = reference_state['psi'].reshape(reference_state['states'], 2, -1)
            target = {'Rho': np.sum(reference_state['attributes'][0, :, None] * np.sum(np.abs(basis)**2, axis=1), axis=0)}
        errors = {key: difference(current[key], target[key]) for key in current}
        for key, error in errors.items():
            assert error['max_abs'] < 1e-9 + 1e-10*np.max(np.abs(target[key])), (key, error)
        row = dict(variant=name, threads=threads, mprint=prints,settings=settings,directory=str(path),input_sha256=sha256(path/'for005'), fields=errors,passed=False)
        rows.append(row)
        save()
        if prints:
            geometry=np.loadtxt(path/'conver.res',ndmin=2)[:,5:11]
            if geometry_reference is None:
                geometry_reference=geometry
            row['geometry']=difference(geometry,geometry_reference)
            save()
            assert np.allclose(geometry,geometry_reference,rtol=1e-9,atol=2e-7), ('Static printed geometry',settings,row['geometry'])
        row['passed']=True
        save()
        print(path.name, 'largest field absolute error', max(e['max_abs'] for e in errors.values()), flush=True)
    save(True)


if __name__ == '__main__':
    main()
