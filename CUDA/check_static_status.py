#!/usr/bin/env python3
"""Exercise static success/exhaustion reports and compare unchanged wavefunctions."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tempfile

import numpy as np

from benchmark import REPO, checkpoint, difference
from static_convergence import TEMPLATE, assign, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True, help='Pre-warning strict CPU executable')
    parser.add_argument('--output', type=Path, default=REPO/'CUDA/results/static-status.json')
    args = parser.parse_args()
    root = Path(tempfile.mkdtemp(prefix='sky3d-static-status-', dir=Path.home()/'.cache'))
    rows = []
    for converged, limit, tolerance in [(False, 3, '1D-6'), (True, 10, '100D0')]:
        for prints in (0, 10):
            text = TEMPLATE.read_text()
            for key, value in [('maxiter', limit), ('serr', tolerance), ('mprint', prints), ('mplot', 0), ('mrest', 3)]:
                text = assign(text, key, value)
            cases = []
            for label, executable in [('current', args.executable), ('baseline', args.baseline)]:
                case = root/f'{converged}-{prints}-{label}'
                case.mkdir()
                (case/'for005').write_text(text)
                run(executable.resolve(), case, 8)
                cases.append(case)
            log = (cases[0]/'stdout.log').read_text()
            if converged:
                assert 'Static convergence criterion met at iteration 2.' in log
                assert 'WARNING: static iteration limit' not in log
                expected_step = 2
            else:
                assert 'WARNING: static iteration limit 3 reached without meeting serr.' in log
                assert 'Static convergence criterion met' not in log
                expected_step = 3
            match = re.search(r'Final pre-gradient fluctuations \(MeV\): h\*\*2=\s*(\S+), h\*h=\s*(\S+), serr=\s*(\S+)', log)
            assert match, cases[0]
            metrics = tuple(map(float, match.groups()))
            actual = checkpoint(cases[0]/'20ne_sly5.tdhf')
            baseline = checkpoint(cases[1]/'20ne_sly5.tdhf')
            assert actual['step'] == baseline['step'] == expected_step
            saved = float(np.dot(actual['attributes'][0], actual['attributes'][5])/actual['states'])
            assert np.isclose(saved, metrics[0], rtol=5e-9, atol=1e-14)
            assert (saved < metrics[2]) == converged
            error = difference(actual['psi'], baseline['psi'])
            assert error['max_abs'] < 1e-12, error
            assert np.allclose(actual['particles'], [10, 10], rtol=0, atol=1e-12)
            rows.append(dict(converged=converged, mprint=prints, saved_iteration=actual['step'],
                             h_squared_fluctuation_mev=metrics[0], h_h_fluctuation_mev=metrics[1],
                             requested_serr=metrics[2], wavefunction_vs_baseline=error))
    result = dict(passed=True, run_directory=str(root), cases=rows,
                  executable_sha256=hashlib.sha256(args.executable.read_bytes()).hexdigest(),
                  baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print('Passed static status and unchanged-wavefunction checks:', args.output)


if __name__ == '__main__':
    main()
