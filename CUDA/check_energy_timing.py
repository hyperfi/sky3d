#!/usr/bin/env python3
"""Measure the legacy diagnostic Coulomb time level without changing evolution."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from benchmark import REPO, checkpoint, densities


def isolated_potential(rho, spacing):
    """Reproduce Code/coulomb.f90's doubled-grid convolution, including origin."""
    grid = rho.shape
    distance = [spacing[a] * np.fft.fftfreq(2*n) * (2*n) for a, n in enumerate(grid)]
    radius2 = (distance[0][:, None, None]**2 + distance[1][None, :, None]**2
               + distance[2][None, None, :]**2)
    radius2[0, 0, 0] = 1
    kernel = 1 / np.sqrt(radius2)
    # Use the implementation, which differs from its anisotropic-grid comment.
    kernel[0, 0, 0] = 2.84 * 3 / sum(spacing)
    padded = np.zeros(tuple(2*n for n in grid))
    crop = tuple(slice(0, n) for n in grid)
    padded[crop] = rho
    return (1.43989 * np.prod(spacing)
            * np.fft.ifftn(np.fft.fftn(padded) * np.fft.fftn(kernel)).real[crop])


def correction(path, state):
    fields = densities(path)
    rho = fields['Rho'].reshape((*state['grid'], 2), order='F')[..., 1]
    stored = fields['Wcoul'].reshape(state['grid'], order='F')
    fresh = isolated_potential(rho, state['spacing'])
    delta = fresh - stored
    return dict(potential_max_abs_difference_mev=float(np.max(np.abs(delta))),
                direct_coulomb_energy_correction_mev=float(0.5 * np.prod(state['spacing']) * np.sum(rho * delta)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.validation.read_text())
    if not evidence.get('passed'):
        parser.error('A completed, passed extended validation report is required')
    expected = evidence['build']['source_sha256']
    for name in ('Code/coulomb.f90', 'Code/params.f90', 'Code/dynamic.f90', 'Code/energies.f90'):
        if hashlib.sha256((REPO / name).read_bytes()).hexdigest() != expected[name]:
            parser.error(f'Source changed since this evidence: {name}')
    if not re.search(r'\be2\s*=\s*1\.43989D0\b', (REPO / 'Code/params.f90').read_text(), re.I):
        parser.error('Coulomb constant changed; update this analysis explicitly')
    rows = []
    for case in evidence['cases']:
        for backend, run in case['runs'].items():
            path = Path(run['directory'])
            if not re.search(r'\bperiodic\s*=\s*F\b', (path / 'for005').read_text(), re.I):
                parser.error('This analysis requires isolated Coulomb')
            state = checkpoint(path / 'k0_restart.tdhf')
            plots = sorted(path.glob('*.tdd'))
            if len(plots) != 2:
                parser.error(f'Expected initial and final density snapshots: {path}')
            first, last = correction(plots[0], state), correction(plots[-1], state)
            if first['potential_max_abs_difference_mev'] > 1e-12:
                raise AssertionError(f'Independent solver disagrees at time zero: {path}: {first}')
            energies = np.loadtxt(path / 'energies.res', ndmin=2)
            printed_drift = float(energies[-1, 4] - energies[0, 4])
            rows.append(dict(case=case['case'], backend=backend, initial=first, final=last,
                             printed_endpoint_energy_drift_mev=printed_drift,
                             corrected_endpoint_energy_drift_mev=(printed_drift
                                 + last['direct_coulomb_energy_correction_mev']
                                 - first['direct_coulomb_energy_correction_mev'])))
    report = dict(
        validation_sha256=hashlib.sha256(args.validation.read_bytes()).hexdigest(),
        state_sha256=evidence['state_sha256'],
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_sha256={name: expected[name] for name in
                       ('Code/coulomb.f90', 'Code/params.f90', 'Code/dynamic.f90', 'Code/energies.f90')},
        interpretation='tinfo precedes the end-of-step skyrme refresh; its direct Coulomb term uses the preceding field',
        limits='Only the direct Coulomb contribution is corrected; printed energies are rounded to 1e-7 MeV. '
               'This does not change the propagator, correct the single-particle energy or establish production accuracy.',
        initial_solver_equivalence_atol_mev=1e-12, cases=rows, passed=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output)


if __name__ == '__main__':
    main()
