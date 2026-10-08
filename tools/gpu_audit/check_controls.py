#!/usr/bin/env python3
"""Regression checks for repaired CPU controls using an isolated WSL build."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import struct
import tempfile

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exe', type=Path, required=True)
    args = parser.parse_args()
    exe = args.exe.resolve()
    assert exe.is_file(), exe
    root = Path(tempfile.mkdtemp(prefix='sky3d-controls-', dir=Path.home() / '.cache'))
    (root / 'initial.tdhf').symlink_to(REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/static/20ne_sly5_zaxis.tdhf')
    text = (REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/td_k0.in').read_text()
    text = text.replace('nt=30000', 'nt=2').replace('mprint=10', 'mprint=1')
    text = text.replace('../../static/20ne_sly5_zaxis.tdhf', '../initial.tdhf')
    cases = {
        'baseline': text,
        'resetcm_every_step': text.replace('mxpact=4,', 'mxpact=4, mrescm=1,'),
        'zero_restart_interval': text.replace('mrest=10000', 'mrest=0'),
        'zero_plot_interval': text.replace('mplot=10000', 'mplot=0'),
        'zero_print_interval': text.replace('mprint=1', 'mprint=0'),
        'long_fragment_path': text.replace('../initial.tdhf', str((root / 'initial.tdhf').resolve())),
        'restart_every_step': text.replace('mrest=10000', 'mrest=1'),
    }
    rows = []
    for name, input_text in cases.items():
        path = root / name
        path.mkdir()
        (path / 'for005').write_text(input_text)
        with (path / 'stdout.log').open('w') as log:
            result = subprocess.run([str(exe)], cwd=path, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        output = path / 'energies.res'
        lines = [l.split() for l in output.read_text().splitlines() if l.strip() and not l.startswith('#')] if output.exists() else []
        row = dict(case=name, returncode=result.returncode, run_directory=str(path),
                   neutron_proton_counts=[[float(l[0]), float(l[1]), float(l[2])] for l in lines],
                   diagnostic_tail=(path / 'stdout.log').read_text()[-1600:])
        rows.append(row)
        print(name, result.returncode, row['neutron_proton_counts'], flush=True)
    baseline, reset, zero_restart, zero_plot, zero_print, long_path, frequent_restart = rows
    assert baseline['returncode'] == 0
    assert reset['returncode'] == 0
    for row in (baseline, reset, zero_restart, zero_plot, long_path):
        assert row['returncode'] == 0, row['case']
        for _, neutrons, protons in row['neutron_proton_counts']:
            assert abs(neutrons - 10.0) < 1e-6 and abs(protons - 10.0) < 1e-6, row['case']
    assert zero_print['returncode'] == 0
    # Initialization always writes a state; mrest=0 must leave its iteration at 0.
    def restart_iteration(row):
        with (Path(row['run_directory']) / 'k0_restart.tdhf').open('rb') as handle:
            return struct.unpack('<ii', handle.read(8))[1]
    assert restart_iteration(zero_restart) == 0
    assert restart_iteration(frequent_restart) == 2
    # Static restart and printing intervals use the same safe predicate.
    path = root / 'static_zero_intervals'
    path.mkdir()
    static = (REPO / 'projects/icnpa2026_20Ne_Kresolved_E2/inputs/static_20ne_sly5.in').read_text()
    static = static.replace('maxiter=3000', 'maxiter=3').replace('mrest=200', 'mrest=0').replace('mprint=10', 'mprint=0')
    (path / 'for005').write_text(static)
    with (path / 'stdout.log').open('w') as log:
        result = subprocess.run([str(exe)], cwd=path, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
    assert result.returncode == 0
    rows.append(dict(case='static_zero_intervals', returncode=result.returncode, run_directory=str(path)))
    # mplot is correctly protected by its outer nonzero-interval check.
    assert zero_plot['returncode'] == 0
    (Path(__file__).parent / 'evidence/controls-fixed.json').write_text(json.dumps(
        dict(date='2026-10-08', source_executable=str(exe), cases=rows), indent=2) + '\n')


if __name__ == '__main__':
    main()
