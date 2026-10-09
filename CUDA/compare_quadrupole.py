#!/usr/bin/env python3
"""Reproduce the archived 20Ne K=0 response on current CPU/GPU executables."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np

from benchmark import REPO, checkpoint, difference, input_text, run
from validate_extended import sha256, verify_build
sys.path.insert(0, str(REPO / 'Utils/Strength_Calculation'))
from sky3d_response.analysis import analyze_signal
from sky3d_response.models import ChannelSpec, WindowSpec

PROJECT = REPO / 'projects/icnpa2026_20Ne_Kresolved_E2'


def prefix(path, end):
    values = np.loadtxt(path, ndmin=2)
    result = values[values[:, 0] <= end + 1e-8]
    if abs(result[0, 0]) > 1e-8 or abs(result[-1, 0] - end) > 1e-8:
        raise ValueError(f'Missing exact [0,{end}] interval: {path}')
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite signal: {path}')
    return result


def signal_error(actual, reference):
    if actual.shape != reference.shape or not np.allclose(actual[:, 0], reference[:, 0], rtol=0, atol=1e-8):
        raise ValueError('Time histories must use identical physical sample times')
    a, b = actual[:, 1] - actual[0, 1], reference[:, 1] - reference[0, 1]
    return dict(raw_max_abs_fm2=float(np.max(np.abs(actual[:, 1] - reference[:, 1]))),
                baseline_subtracted=difference(a, b))


def transform(values, reference, gamma=0.5):
    if values.shape != reference.shape or not np.allclose(values[:, 0], reference[:, 0], rtol=0, atol=1e-8):
        raise ValueError('Unboosted reference requires matching physical sample times')
    return analyze_signal(values[:, 0], values[:, 1],
                          ChannelSpec('K0_code_native', Path('quadrupoles.res'), 1, 2,
                                      strength_units='fm^4 / MeV'),
                          WindowSpec(f'gamma_{gamma}', gamma_mev=gamma), boost_amplitude=5e-5,
                          zero_padding_factor=8, baseline_points=1, reference_signal=reference[:, 1])


def spectral_error(actual, reference, energy, lo=0.5, hi=35):
    mask = (energy >= lo) & (energy <= hi)
    error = difference(actual[mask], reference[mask])
    error['max_abs_over_reference_peak'] = error['max_abs'] / float(np.max(np.abs(reference[mask])))
    return error


def plots(destination, signals, spectra, smoothing_spectra):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.facecolor': 'white', 'axes.titleweight': 'bold'})
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.6), constrained_layout=True)
    styles = {'CPU': ('#205593', '-', 2.0), 'GPU': ('#da671e', '--', 1.8),
              'Saved CPU': ('#333333', ':', 1.2)}
    for label in ('CPU', 'GPU', 'Saved CPU'):
        color, style, width = styles[label]
        spectrum = spectra[label]
        axes[0].plot(spectrum.time_fm_c, spectrum.delta_signal, color=color,
                     ls=style, lw=width, label=label)
        mask = (spectrum.energy_mev >= 0) & (spectrum.energy_mev <= 35)
        axes[1].plot(spectrum.energy_mev[mask], spectrum.signed_strength[mask], color=color,
                     ls=style, lw=width, label=label)
    axes[0].set(xlabel='Time (fm/c)', ylabel=r'$\delta\langle F_{20}\rangle$ (fm$^2$)',
                title=r'$^{20}$Ne quadrupole time response, K=0')
    axes[1].set(xlabel='Energy (MeV)', ylabel=r'$S(E)$ (fm$^4$/MeV)',
                title=r'Isoscalar quadrupole strength · $\Gamma_{sm}=0.5$ MeV')
    axes[1].set_xlim(0, 35)
    for axis in axes:
        axis.legend(frameon=False, ncol=3)
        axis.grid(alpha=0.18)
    fig.suptitle('SLy5 · 24³ / 1 fm · dt=0.2 fm/c · 6000 fm/c · identical aligned state', fontsize=12)
    fig.savefig(destination / 'quadrupole-overlay.png', dpi=170)
    fig.savefig(destination / 'quadrupole-overlay.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), constrained_layout=True)
    cpu, gpu, saved = (spectra[label] for label in ('CPU', 'GPU', 'Saved CPU'))
    axes[0].plot(cpu.time_fm_c, gpu.delta_signal-cpu.delta_signal, color='#da671e', label='GPU − current CPU')
    axes[0].plot(cpu.time_fm_c, gpu.delta_signal-saved.delta_signal, color='#333333', ls=':', label='GPU − saved CPU')
    axes[0].set(xlabel='Time (fm/c)', ylabel=r'Difference (fm$^2$)', title='Time-response differences')
    mask = (cpu.energy_mev >= 0.5) & (cpu.energy_mev <= 35)
    scale = np.max(np.abs(saved.signed_strength[mask]))
    axes[1].plot(cpu.energy_mev[mask], (gpu.signed_strength-cpu.signed_strength)[mask]/scale,
                 color='#da671e', label='GPU − current CPU')
    axes[1].plot(cpu.energy_mev[mask], (gpu.signed_strength-saved.signed_strength)[mask]/scale,
                 color='#333333', ls=':', label='GPU − saved CPU')
    axes[1].set(xlabel='Energy (MeV)', ylabel='Difference / saved peak', title='Strength differences without fitting or rescaling')
    for axis in axes:
        axis.legend(frameon=False)
        axis.grid(alpha=0.18)
        axis.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    fig.savefig(destination / 'quadrupole-differences.png', dpi=170)
    fig.savefig(destination / 'quadrupole-differences.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True, sharex=True)
    for axis, gamma in zip(axes, (1.0, 0.5, 0.2)):
        for label, spectrum in smoothing_spectra[gamma].items():
            color, style, width = styles[label]
            mask = spectrum.energy_mev <= 35
            axis.plot(spectrum.energy_mev[mask], spectrum.signed_strength[mask],
                      color=color, ls=style, lw=width, label=label)
        axis.set(title=rf'$\Gamma_{{sm}}={gamma:g}$ MeV', ylabel=r'$S(E)$ (fm$^4$/MeV)')
        axis.legend(frameon=False, ncol=3)
        axis.grid(alpha=0.18)
    axes[-1].set(xlabel='Energy (MeV)', xlim=(0, 35))
    fig.suptitle(r'$^{20}$Ne K=0 · all three smoothing widths used in the saved plot', fontsize=12)
    fig.savefig(destination / 'quadrupole-smoothing-overlay.png', dpi=170)
    fig.savefig(destination / 'quadrupole-smoothing-overlay.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path, help='Analyze an existing completed run instead of running again')
    parser.add_argument('--particle-atol', type=float, default=1e-6,
                        help='Explicit replay-only absolute particle budget; fine-grid tests retain 1e-6')
    args = parser.parse_args()
    metadata, executables = verify_build(args.build_dir.resolve())
    if not np.isfinite(args.particle_atol) or args.particle_atol <= 0:
        parser.error('Particle tolerance must be finite and positive')
    state_path = PROJECT / 'static/20ne_sly5_zaxis.tdhf'
    state = checkpoint(state_path)
    if (state['grid'] != (24, 24, 24) or state['spacing'] != (1, 1, 1)
            or state['nucleons'] != (10, 10) or state['force'].lower() != 'sly5'):
        parser.error('Archived aligned Sly5 20Ne / 24³ / 1 fm state required')
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / 'quadrupole-comparison.json'
    previous = json.loads(report_path.read_text()) if args.run_directory and report_path.exists() else None
    if args.run_directory and previous is None:
        parser.error('Reanalysis requires the recorded comparison report and build provenance')
    if args.run_directory and previous and previous['build']['executable_sha256'] != metadata['executable_sha256']:
        parser.error('Analysis build differs from the recorded run')
    if args.run_directory and previous and Path(previous['run_directory']).resolve() != args.run_directory.resolve():
        parser.error('Run directory differs from the recorded run')
    root = args.run_directory or Path(tempfile.mkdtemp(prefix='sky3d-quadrupole-', dir=Path.home() / '.cache'))
    report = dict(run_directory=str(root), state_sha256=sha256(state_path), build=metadata,
                  runner_sha256=sha256(Path(__file__)),
                  execution_runner_sha256=(previous.get('execution_runner_sha256', previous['runner_sha256'])
                                           if previous else sha256(Path(__file__))),
                  physics=dict(nucleus='20Ne', force='Sly5', grid=[24]*3, spacing_fm=[1]*3,
                               dt_fm_c=0.2, steps=30000, endpoint_fm_c=6000, amplitude=5e-5,
                               L=2, M=0, intrinsic_axis='z', gamma_sm_mev=0.5),
                  reference_note='All spectra use the same archived unboosted CPU reference; fresh 200 fm/c CPU/GPU controls check this reference.',
                  tolerances=dict(raw_Q_max_abs_fm2=1e-6, signal_relative_l2=1e-5,
                                  spectrum_relative_l2=1e-5, spectrum_max_abs_over_peak=1e-5),
                  runs=previous.get('runs', {}) if previous else {}, passed=False)

    def save():
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    archived = PROJECT / 'response/prefix_runs'
    old_signal = prefix(archived / 'K0/quadrupoles.res', 6000)
    old_reference = prefix(archived / 'reference_K0/quadrupoles.res', 6000)
    report['tolerances']['checkpoint_particle_atol'] = args.particle_atol
    report['archived_final_checkpoint_particles'] = checkpoint(PROJECT / 'td/K0/k0_restart.tdhf')['particles']
    report['archived_sha256'] = {str(p.relative_to(REPO)): sha256(p) for p in
        (archived / 'K0/quadrupoles.res', archived / 'reference_K0/quadrupoles.res',
         PROJECT / 'response/K0/spectra.csv', PROJECT / 'inputs/response_k0.toml')}
    save()
    try:
        if not args.run_directory:
            shutil.copy2(state_path, root / 'initial.tdhf')
            # Calibrate the shared unboosted reference with current backends.
            for backend in ('cpu', 'gpu'):
                text = input_text(1000, amplitude='0.0D0')
                text = text.replace('texternal=T', 'texternal=F')
                path = root / f'unboosted-{backend}'
                report['runs'][f'unboosted-{backend}'] = run(executables[backend], path, text, 8, backend)
                actual = prefix(path / 'quadrupoles.res', 200)
                report['runs'][f'unboosted-{backend}']['archived_trace'] = signal_error(actual, prefix(archived / 'reference_K0/quadrupoles.res', 200))
                save()
            for backend in ('cpu', 'gpu'):
                path = root / backend
                text = input_text(30000)  # same pulse, timestep, order and print interval as the archive
                report['runs'][backend] = run(executables[backend], path, text, 8, backend,
                                             particle_atol=args.particle_atol)
                save()
        else:
            if sha256(root / 'initial.tdhf') != report['state_sha256']:
                raise AssertionError('Run seed differs from the recorded aligned state')
            for backend in ('cpu', 'gpu'):
                final = checkpoint(root / backend / 'k0_restart.tdhf')
                if final['step'] != 30000 or abs(final['time'] - 6000) > 1e-8:
                    raise AssertionError(f'Incomplete {backend} checkpoint')
                if max(abs(p - 10) for p in final['particles']) >= args.particle_atol:
                    raise AssertionError(f'{backend} checkpoint particle drift exceeds the replay budget')
        signals = {'CPU': prefix(root / 'cpu/quadrupoles.res', 6000),
                   'GPU': prefix(root / 'gpu/quadrupoles.res', 6000), 'Saved CPU': old_signal}
        report['time_comparisons'] = dict(
            gpu_vs_current_cpu=signal_error(signals['GPU'], signals['CPU']),
            gpu_vs_saved_cpu=signal_error(signals['GPU'], old_signal),
            current_cpu_vs_saved_cpu=signal_error(signals['CPU'], old_signal))
        for backend in ('cpu', 'gpu'):
            control = report['runs'].get(f'unboosted-{backend}', {}).get('archived_trace')
            if control and control['raw_max_abs_fm2'] > 1e-6:
                raise AssertionError(f'Unboosted archive control differs: {control}')
        spectra = {label: transform(values, old_reference) for label, values in signals.items()}
        energy = spectra['CPU'].energy_mev
        strength = {label: item.signed_strength for label, item in spectra.items()}
        report['spectrum_comparisons'] = dict(
            gpu_vs_current_cpu=spectral_error(strength['GPU'], strength['CPU'], energy),
            gpu_vs_saved_cpu=spectral_error(strength['GPU'], strength['Saved CPU'], energy),
            current_cpu_vs_saved_cpu=spectral_error(strength['CPU'], strength['Saved CPU'], energy))
        report['spectrum_comparisons_gqr'] = dict(
            gpu_vs_current_cpu=spectral_error(strength['GPU'], strength['CPU'], energy, 10, 35),
            gpu_vs_saved_cpu=spectral_error(strength['GPU'], strength['Saved CPU'], energy, 10, 35))
        report['final_wavefunction_cpu_gpu'] = difference(
            checkpoint(root / 'gpu/k0_restart.tdhf')['psi'], checkpoint(root / 'cpu/k0_restart.tdhf')['psi'])
        if report['final_wavefunction_cpu_gpu']['relative_l2'] > 1e-8:
            raise AssertionError('Long-run CPU/GPU wavefunctions differ')
        saved_rows = list(csv.DictReader((PROJECT / 'response/K0/spectra.csv').open()))
        rows = [row for row in saved_rows if row['window'] == 'gamma_0p5']
        saved_energy = np.array([float(row['energy_mev']) for row in rows])
        saved_strength = np.array([float(row['signed_strength']) for row in rows])
        indices = np.flatnonzero(energy <= 35)
        if len(indices) != len(rows) or not np.allclose(energy[indices], saved_energy, rtol=0, atol=1e-9):
            raise AssertionError('Archived spectrum energy grid differs; do not interpolate silently')
        report['archived_analysis_reproduction'] = difference(strength['Saved CPU'][indices], saved_strength)
        if report['archived_analysis_reproduction']['relative_l2'] > 1e-9:
            raise AssertionError('Analysis pipeline does not reproduce the saved spectrum')
        smoothing_spectra = {}
        report['all_smoothing_comparisons'] = {}
        smoothing_failures = []
        for gamma, name in ((1.0, 'gamma_1p0'), (0.5, 'gamma_0p5'), (0.2, 'gamma_0p2')):
            items = spectra if gamma == 0.5 else {label: transform(values, old_reference, gamma)
                                                for label, values in signals.items()}
            smoothing_spectra[gamma] = items
            window_rows = [row for row in saved_rows if row['window'] == name]
            e = np.array([float(row['energy_mev']) for row in window_rows])
            s = np.array([float(row['signed_strength']) for row in window_rows])
            if len(e) != len(indices) or not np.allclose(e, energy[indices], rtol=0, atol=1e-9):
                raise AssertionError('Saved smoothing energy grid differs')
            errors = dict(gpu_vs_current_cpu=spectral_error(items['GPU'].signed_strength,
                          items['CPU'].signed_strength, energy),
                          gpu_vs_saved_cpu=spectral_error(items['GPU'].signed_strength,
                          items['Saved CPU'].signed_strength, energy),
                          archived_analysis_reproduction=difference(items['Saved CPU'].signed_strength[indices], s))
            if errors['archived_analysis_reproduction']['relative_l2'] > 1e-9:
                raise AssertionError(f'Cannot reproduce the saved {name} curve')
            for key in ('gpu_vs_current_cpu', 'gpu_vs_saved_cpu'):
                if errors[key]['relative_l2'] > 1e-5 or errors[key]['max_abs_over_reference_peak'] > 1e-5:
                    smoothing_failures.append(f'{name}: {key}: {errors[key]}')
            report['all_smoothing_comparisons'][name] = errors
        gqr = np.flatnonzero((energy >= 10) & (energy <= 35))
        report['gqr_peaks_mev'] = {label: float(energy[gqr[np.argmax(values[gqr])]]) for label, values in strength.items()}
        report['rayleigh_resolution_mev'] = spectra['CPU'].rayleigh_resolution_mev
        plots(args.output, signals, spectra, smoothing_spectra)
        if smoothing_failures:
            raise AssertionError('; '.join(smoothing_failures))
        for error in report['time_comparisons'].values():
            if error['raw_max_abs_fm2'] > 1e-6 or error['baseline_subtracted']['relative_l2'] > 1e-5:
                raise AssertionError(f'Time comparison failed: {error}')
        for error in [*report['spectrum_comparisons'].values(), *report['spectrum_comparisons_gqr'].values()]:
            if error['relative_l2'] > 1e-5 or error['max_abs_over_reference_peak'] > 1e-5:
                raise AssertionError(f'Spectrum comparison failed: {error}')
        report['passed'] = True
        save()
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
        save()
        raise
    print(f'Quadrupole comparison passed: {report_path}; raw output: {root}', flush=True)


if __name__ == '__main__':
    main()
