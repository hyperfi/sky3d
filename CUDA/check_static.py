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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, default=REPO / 'CUDA/build')
    parser.add_argument('--output', type=Path, default=REPO / 'CUDA/results/static.json')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='sky3d-static-check-', dir=Path.home() / '.cache'))
    text = (REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/static_20ne_sly5.in').read_text()
    text = text.replace('maxiter=3000', 'maxiter=120').replace('mrest=200', 'mrest=120').replace('mplot=100000', 'mplot=120')
    text = text.replace("writeselect='r'", "writeselect='rtcsouw', write_isospin=T")
    rows = []
    reference = None
    reference_state = None
    for name, threads, prints in [('reference', 1, 10), ('reference', 8, 10), ('reference', 8, 10), ('cpu', 8, 10), ('cpu', 8, 0), ('gpu', 8, 10), ('gpu', 8, 0)]:
        path = root / f'{len(rows)}-{name}-{threads}'
        path.mkdir()
        (path / 'for005').write_text(text.replace('mprint=10', f'mprint={prints}'))
        exe = args.build_dir.resolve() / name / f'sky3d.{name}'
        with (path / 'stdout.log').open('w') as log:
            subprocess.run([str(exe)], cwd=path, check=True, stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, SKY3D_BACKEND='gpu' if name == 'gpu' else 'cpu', OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS='1'))
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
        row = dict(variant=name, threads=threads, mprint=prints, directory=str(path), fields=errors)
        rows.append(row)
        print(path.name, 'largest field absolute error', max(e['max_abs'] for e in errors.values()), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(run_directory=str(root), steps=120, passed=True, cases=rows,
         build=json.loads((args.build_dir/'build.json').read_text()),
         note='Full static density/potential comparison, not individual degenerate-orbital phases or convergence proof'), indent=2) + '\n')


if __name__ == '__main__':
    main()
