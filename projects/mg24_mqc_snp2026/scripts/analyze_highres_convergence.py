#!/usr/bin/env python3
"""Analyze long-time, smoothing, stability, density-edge, and reciprocity convergence."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sky3d_response.analysis import analyze_signal, calculate_summary
from sky3d_response.constants import HBARC_MEV_FM
from sky3d_response.density import density_diagnostics, read_tdd_density
from sky3d_response.io import read_response_file
from sky3d_response.models import ChannelSpec, RegionSpec, Spectrum, WindowSpec


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed" / "highres"
FIGURES = PROJECT / "figures"
ETA = 5.0e-5
DURATIONS = (2000.0, 4000.0, 8000.0, 12000.0, 18000.0)
GAMMAS = (1.0, 0.5, 0.2, 0.1, 0.07)
ZERO_PADDING = 8
WINDOW_END_THRESHOLD = 0.05
SPECTRUM_RANGE = (8.0, 26.0)
REGIONS = (
    RegionSpec("mqc", 9.0, 18.0),
    RegionSpec("main_isgmr", 18.0, 25.0),
    RegionSpec("experiment_range", 9.0, 25.0),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--sly5-e0", type=Path, required=True)
    result.add_argument("--sly5-reference", type=Path, required=True)
    result.add_argument("--skms-e0", type=Path, required=True)
    result.add_argument("--skms-reference", type=Path, required=True)
    result.add_argument("--sly5-e2", type=Path)
    result.add_argument("--skms-e2", type=Path)
    result.add_argument("--sly5-half", type=Path)
    result.add_argument("--skms-half", type=Path)
    result.add_argument("--sly5-large-e0", type=Path)
    result.add_argument("--sly5-large-reference", type=Path)
    result.add_argument("--skms-large-e0", type=Path)
    result.add_argument("--skms-large-reference", type=Path)
    result.add_argument("--output", type=Path, default=PROCESSED)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_table(path: Path) -> np.ndarray:
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    duplicate = np.flatnonzero(np.diff(data[:, 0]) == 0.0)
    for index in duplicate:
        if not np.allclose(data[index, 1:], data[index + 1, 1:], rtol=1.0e-8, atol=1.0e-10):
            raise ValueError(f"{path}: inconsistent duplicate restart point")
    if duplicate.size:
        keep = np.ones(len(data), dtype=bool)
        keep[duplicate] = False
        data = data[keep]
    if np.any(np.diff(data[:, 0]) <= 0):
        raise ValueError(f"{path}: first column must increase strictly")
    return data


def restrict(time: np.ndarray, values: np.ndarray, duration: float) -> tuple[np.ndarray, np.ndarray]:
    mask = time <= duration + 1.0e-8
    selected_time = time[mask]
    selected_values = values[mask]
    if not len(selected_time) or not np.isclose(selected_time[-1], duration, atol=1.0e-8):
        available = float(time[-1]) if len(time) else float("nan")
        raise ValueError(f"requested T={duration:g} fm/c, but final available time is {available:g}")
    return selected_time, selected_values


def ewsr_references(reference_run: Path, edf: str) -> tuple[float, float]:
    mono = numeric_table(reference_run / "monopoles.res")[0]
    quad = numeric_table(reference_run / "quadrupoles.res")[0]
    sqrt_4pi = math.sqrt(4.0 * math.pi)
    mono_n = 0.5 * (mono[1] - mono[2])
    mono_p = 0.5 * (mono[1] + mono[2])
    r2_n = mono_n * sqrt_4pi
    r2_p = mono_p * sqrt_4pi
    q20_n = 0.5 * (quad[1] - quad[2])
    q20_p = 0.5 * (quad[1] + quad[2])
    if edf == "SLy5":
        h2m_n = h2m_p = 20.7355298
    elif edf == "SkM*":
        h2m_n = h2m_p = 20.7525
    else:
        raise ValueError(f"unknown EDF {edf!r}")
    e0 = (h2m_n * r2_n + h2m_p * r2_p) / math.pi
    c20 = 5.0 / (4.0 * math.sqrt(math.pi))
    grad2_n = 25.0 / (4.0 * math.pi) * (2.0 * r2_n + q20_n / c20)
    grad2_p = 25.0 / (4.0 * math.pi) * (2.0 * r2_p + q20_p / c20)
    e2 = h2m_n * grad2_n + h2m_p * grad2_p
    return float(e0), float(e2)


def load_observable(run: Path, filename: str, column: int = 1) -> tuple[np.ndarray, np.ndarray]:
    time, signal, _ = read_response_file(run / filename, column)
    return time, signal


def windows() -> tuple[WindowSpec, ...]:
    return tuple(
        WindowSpec(f"gamma_{gamma:g}".replace(".", "p"), "exponential", gamma)
        for gamma in GAMMAS
    )


def channels(e0_ewsr: float, e2_ewsr: float, mode: str) -> tuple[ChannelSpec, ...]:
    if mode == "e0":
        return (
            ChannelSpec(
                name="e0_diagonal",
                file=Path("monopoles.res"),
                column=1,
                multipolarity=0,
                projection=0,
                channel_type="isoscalar",
                diagonal=True,
                strength_units="fm^4 / MeV",
                ewsr_reference=e0_ewsr,
                ewsr_label="IS E0 double-commutator EWSR in Sky3D F00 convention",
            ),
            ChannelSpec(
                name="q20_from_e0",
                file=Path("quadrupoles.res"),
                column=1,
                multipolarity=2,
                projection=0,
                channel_type="cross",
                diagonal=False,
                strength_units="fm^4 / MeV",
            ),
        )
    if mode == "e2":
        return (
            ChannelSpec(
                name="e2k0_diagonal",
                file=Path("quadrupoles.res"),
                column=1,
                multipolarity=2,
                projection=0,
                channel_type="isoscalar",
                diagonal=True,
                strength_units="fm^4 / MeV",
                ewsr_reference=e2_ewsr,
                ewsr_label="IS E2 K=0 double-commutator EWSR in Sky3D F20 convention",
            ),
            ChannelSpec(
                name="q00_from_e2k0",
                file=Path("monopoles.res"),
                column=1,
                multipolarity=0,
                projection=0,
                channel_type="cross",
                diagonal=False,
                strength_units="fm^4 / MeV",
            ),
        )
    raise ValueError(mode)


def analyze_family(
    *,
    family: str,
    edf: str,
    run: Path,
    reference_run: Path,
    mode: str,
    durations: Iterable[float],
    boost_amplitude: float,
) -> list[Spectrum]:
    e0_ewsr, e2_ewsr = ewsr_references(reference_run, edf)
    result: list[Spectrum] = []
    reference_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel in channels(e0_ewsr, e2_ewsr, mode):
        time, signal = load_observable(run, channel.file.name, channel.column)
        if channel.file.name not in reference_cache:
            reference_cache[channel.file.name] = load_observable(
                reference_run, channel.file.name, channel.column
            )
        reference_time, reference_signal = reference_cache[channel.file.name]
        if not np.array_equal(time, reference_time):
            raise ValueError(f"{family}/{channel.name}: boosted and reference time grids differ")
        for duration in durations:
            time_slice, signal_slice = restrict(time, signal, duration)
            _, reference_slice = restrict(reference_time, reference_signal, duration)
            for window in windows():
                result.append(
                    analyze_signal(
                        time_slice,
                        signal_slice,
                        channel,
                        window,
                        boost_amplitude=boost_amplitude,
                        zero_padding_factor=ZERO_PADDING,
                        baseline_points=1,
                        reference_signal=reference_slice,
                        metadata={
                            "family": family,
                            "edf": edf,
                            "mode": mode,
                            "boost_amplitude": boost_amplitude,
                            "run_directory": str(run.resolve()),
                            "reference_directory": str(reference_run.resolve()),
                            "source_sha256": sha256(run / channel.file.name),
                            "reference_sha256": sha256(reference_run / channel.file.name),
                            "zero_padding_interpretation": (
                                "display-grid interpolation only; not physical resolution"
                            ),
                        },
                    )
                )
    return result


def spectrum_key(item: Spectrum) -> tuple[str, float, str, float]:
    return (
        str(item.metadata["family"]),
        item.propagation_time_fm_c,
        item.channel.name,
        float(item.window.gamma_mev),
    )


def summary_rows(spectra: Iterable[Spectrum]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in spectra:
        for region in REGIONS:
            row = asdict(calculate_summary(item, region, nucleus_a=24, nucleus_z=12))
            row.update(
                {
                    "family": item.metadata["family"],
                    "edf": item.metadata["edf"],
                    "mode": item.metadata["mode"],
                    "propagation_time_fm_c": item.propagation_time_fm_c,
                    "rayleigh_resolution_mev": item.rayleigh_resolution_mev,
                    "zero_padded_energy_bin_mev": item.energy_bin_mev,
                    "zero_padding_factor": ZERO_PADDING,
                    "window_end_amplitude": float(item.window_values[-1]),
                    "window_end_below_0p05": bool(
                        item.window_values[-1] <= WINDOW_END_THRESHOLD
                    ),
                }
            )
            rows.append(row)
    return rows


def density_rows(run: Path, family: str, role: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run.glob("*.tdd")):
        row = density_diagnostics(read_tdd_density(path), shell_thickness_fm=2.0)
        row.update({"family": family, "role": role, "file": path.name})
        rows.append(row)
    return rows


def drift_metrics(values: np.ndarray) -> tuple[float, float, float]:
    delta = values - values[0]
    return float(delta[-1]), float(np.max(np.abs(delta))), float(np.ptp(values))


def stability_rows(
    run: Path,
    family: str,
    role: str,
    densities: list[dict[str, object]],
    durations: Iterable[float],
) -> list[dict[str, object]]:
    energy = numeric_table(run / "energies.res")
    mono = numeric_table(run / "monopoles.res")
    quad = numeric_table(run / "quadrupoles.res")
    result: list[dict[str, object]] = []
    for duration in durations:
        energy_slice = energy[energy[:, 0] <= duration + 1.0e-8]
        mono_slice = mono[mono[:, 0] <= duration + 1.0e-8]
        quad_slice = quad[quad[:, 0] <= duration + 1.0e-8]
        if not np.isclose(energy_slice[-1, 0], duration):
            raise ValueError(f"{family}/{role}: missing protocol endpoint T={duration:g}")
        density_slice = [row for row in densities if float(row["time_fm_c"]) <= duration + 1.0e-8]
        final_energy, max_energy, range_energy = drift_metrics(energy_slice[:, 3])
        final_n, max_n, _ = drift_metrics(energy_slice[:, 1])
        final_p, max_p, _ = drift_metrics(energy_slice[:, 2])
        final_q00, max_q00, range_q00 = drift_metrics(mono_slice[:, 1])
        final_q20, max_q20, range_q20 = drift_metrics(quad_slice[:, 1])
        density_n = np.asarray([float(row["particle_number"]) for row in density_slice])
        density_rms = np.asarray([float(row["rms_radius_fm"]) for row in density_slice])
        density_shell = np.asarray([float(row["shell_particles"]) for row in density_slice])
        density_outer = np.asarray(
            [float(row["outermost_max_density_fm3"]) for row in density_slice]
        )
        result.append(
            {
                "family": family,
                "role": role,
                "propagation_time_fm_c": duration,
                "energy_initial_mev": float(energy_slice[0, 3]),
                "energy_final_drift_mev": final_energy,
                "energy_max_abs_drift_mev": max_energy,
                "energy_peak_to_peak_mev": range_energy,
                "neutron_number_final_drift_protocol": final_n,
                "neutron_number_max_abs_drift_protocol": max_n,
                "proton_number_final_drift_protocol": final_p,
                "proton_number_max_abs_drift_protocol": max_p,
                "protocol_particle_number_print_precision": 0.001,
                "q00_initial_fm2": float(mono_slice[0, 1]),
                "q00_final_change_fm2": final_q00,
                "q00_max_abs_change_fm2": max_q00,
                "q00_peak_to_peak_fm2": range_q00,
                "q20_initial_fm2": float(quad_slice[0, 1]),
                "q20_final_change_fm2": final_q20,
                "q20_max_abs_change_fm2": max_q20,
                "q20_peak_to_peak_fm2": range_q20,
                "density_snapshot_count": len(density_slice),
                "density_particle_number_initial": float(density_n[0]),
                "density_particle_number_max_abs_drift": float(
                    np.max(np.abs(density_n - density_n[0]))
                ),
                "rms_radius_initial_fm": float(density_rms[0]),
                "rms_radius_max_abs_change_fm": float(
                    np.max(np.abs(density_rms - density_rms[0]))
                ),
                "boundary_shell_particles_initial": float(density_shell[0]),
                "boundary_shell_particles_max": float(np.max(density_shell)),
                "boundary_shell_particles_max_abs_change": float(
                    np.max(np.abs(density_shell - density_shell[0]))
                ),
                "outermost_max_density_max_fm3": float(np.max(density_outer)),
            }
        )
    return result


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = 0.5 * (np.linalg.norm(first) + np.linalg.norm(second))
    return float(np.linalg.norm(first - second) / denominator) if denominator else float("nan")


def convergence_rows(spectra: list[Spectrum]) -> list[dict[str, object]]:
    lookup = {spectrum_key(item): item for item in spectra}
    rows: list[dict[str, object]] = []
    families = sorted({str(item.metadata["family"]) for item in spectra})
    common_energy = np.linspace(9.0, 25.0, 3201)
    for family in families:
        family_items = [item for item in spectra if item.metadata["family"] == family]
        durations = sorted({item.propagation_time_fm_c for item in family_items})
        reference_duration = max(durations)
        for channel in sorted({item.channel.name for item in family_items}):
            for gamma in GAMMAS:
                reference_key = (family, reference_duration, channel, gamma)
                if reference_key not in lookup:
                    continue
                reference = lookup[reference_key]
                ref_values = np.interp(common_energy, reference.energy_mev, reference.response.real) + 1j * np.interp(
                    common_energy, reference.energy_mev, reference.response.imag
                )
                for duration in durations:
                    key = (family, duration, channel, gamma)
                    if key not in lookup:
                        continue
                    item = lookup[key]
                    values = np.interp(common_energy, item.energy_mev, item.response.real) + 1j * np.interp(
                        common_energy, item.energy_mev, item.response.imag
                    )
                    rows.append(
                        {
                            "family": family,
                            "channel": channel,
                            "gamma_mev": gamma,
                            "propagation_time_fm_c": duration,
                            "reference_time_fm_c": reference_duration,
                            "complex_response_symmetric_relative_l2_9_25": relative_l2(
                                values, ref_values
                            ),
                            "signed_strength_symmetric_relative_l2_9_25": relative_l2(
                                values.imag / math.pi, ref_values.imag / math.pi
                            ),
                        }
                    )
    return rows


def reciprocity_rows(spectra: list[Spectrum]) -> list[dict[str, object]]:
    lookup = {spectrum_key(item): item for item in spectra}
    rows: list[dict[str, object]] = []
    common_energy = np.linspace(9.0, 25.0, 3201)
    for edf_slug, edf in (("sly5", "SLy5"), ("skms", "SkM*")):
        forward_family = f"{edf_slug}_n24_e0"
        reverse_family = f"{edf_slug}_n24_e2"
        durations = sorted(
            {
                item.propagation_time_fm_c
                for item in spectra
                if item.metadata["family"] == forward_family
            }
        )
        for duration in durations:
            for gamma in GAMMAS:
                forward_key = (forward_family, duration, "q20_from_e0", gamma)
                reverse_key = (reverse_family, duration, "q00_from_e2k0", gamma)
                if forward_key not in lookup or reverse_key not in lookup:
                    continue
                forward = lookup[forward_key]
                reverse = lookup[reverse_key]
                a = np.interp(common_energy, forward.energy_mev, forward.response.real) + (
                    1j * np.interp(common_energy, forward.energy_mev, forward.response.imag)
                )
                b = np.interp(common_energy, reverse.energy_mev, reverse.response.real) + (
                    1j * np.interp(common_energy, reverse.energy_mev, reverse.response.imag)
                )
                denominator = float(np.vdot(a, a).real)
                scale = float(np.vdot(a, b).real / denominator) if denominator else float("nan")
                rows.append(
                    {
                        "edf": edf,
                        "propagation_time_fm_c": duration,
                        "gamma_mev": gamma,
                        "complex_response_symmetric_relative_l2_9_25": relative_l2(a, b),
                        "signed_strength_symmetric_relative_l2_9_25": relative_l2(
                            a.imag / math.pi, b.imag / math.pi
                        ),
                        "signed_strength_correlation_9_25": float(
                            np.corrcoef(a.imag, b.imag)[0, 1]
                        ),
                        "best_real_reverse_over_forward_scale": scale,
                    }
                )
    return rows


def linearity_rows(spectra: list[Spectrum]) -> list[dict[str, object]]:
    lookup = {spectrum_key(item): item for item in spectra}
    rows: list[dict[str, object]] = []
    common_energy = np.linspace(9.0, 25.0, 3201)
    for edf_slug, edf in (("sly5", "SLy5"), ("skms", "SkM*")):
        full_family = f"{edf_slug}_n24_e0"
        half_family = f"{edf_slug}_n24_e0_half"
        full_items = [item for item in spectra if item.metadata["family"] == full_family]
        durations = sorted({item.propagation_time_fm_c for item in full_items})
        channels_present = sorted({item.channel.name for item in full_items})
        for duration in durations:
            for channel in channels_present:
                for gamma in GAMMAS:
                    full_key = (full_family, duration, channel, gamma)
                    half_key = (half_family, duration, channel, gamma)
                    if full_key not in lookup or half_key not in lookup:
                        continue
                    full = lookup[full_key]
                    half = lookup[half_key]
                    full_response = np.interp(
                        common_energy, full.energy_mev, full.response.real
                    ) + 1j * np.interp(common_energy, full.energy_mev, full.response.imag)
                    half_response = np.interp(
                        common_energy, half.energy_mev, half.response.real
                    ) + 1j * np.interp(common_energy, half.energy_mev, half.response.imag)
                    rows.append(
                        {
                            "edf": edf,
                            "channel": channel,
                            "propagation_time_fm_c": duration,
                            "gamma_mev": gamma,
                            "full_boost_amplitude": full.boost_amplitude,
                            "half_boost_amplitude": half.boost_amplitude,
                            "time_response_symmetric_relative_l2": relative_l2(
                                full.delta_signal / full.boost_amplitude,
                                half.delta_signal / half.boost_amplitude,
                            ),
                            "complex_response_symmetric_relative_l2_9_25": relative_l2(
                                full_response, half_response
                            ),
                            "signed_strength_symmetric_relative_l2_9_25": relative_l2(
                                full_response.imag / math.pi,
                                half_response.imag / math.pi,
                            ),
                        }
                    )
    return rows


def box_size_rows(spectra: list[Spectrum]) -> list[dict[str, object]]:
    lookup = {spectrum_key(item): item for item in spectra}
    rows: list[dict[str, object]] = []
    common_energy = np.linspace(9.0, 25.0, 3201)
    duration = 8000.0
    for edf_slug, edf in (("sly5", "SLy5"), ("skms", "SkM*")):
        small_family = f"{edf_slug}_n24_e0"
        large_family = f"{edf_slug}_n32_e0"
        for channel in ("e0_diagonal", "q20_from_e0"):
            for gamma in GAMMAS:
                small_key = (small_family, duration, channel, gamma)
                large_key = (large_family, duration, channel, gamma)
                if small_key not in lookup or large_key not in lookup:
                    continue
                small = lookup[small_key]
                large = lookup[large_key]
                small_response = np.interp(
                    common_energy, small.energy_mev, small.response.real
                ) + 1j * np.interp(common_energy, small.energy_mev, small.response.imag)
                large_response = np.interp(
                    common_energy, large.energy_mev, large.response.real
                ) + 1j * np.interp(common_energy, large.energy_mev, large.response.imag)
                rows.append(
                    {
                        "edf": edf,
                        "channel": channel,
                        "propagation_time_fm_c": duration,
                        "gamma_mev": gamma,
                        "small_grid": "24^3",
                        "large_grid": "32^3",
                        "spacing_fm": 1.0,
                        "complex_response_symmetric_relative_l2_9_25": relative_l2(
                            small_response, large_response
                        ),
                        "signed_strength_symmetric_relative_l2_9_25": relative_l2(
                            small_response.imag / math.pi,
                            large_response.imag / math.pi,
                        ),
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_spectrum_products(output: Path, spectra: list[Spectrum]) -> None:
    spectrum_rows: list[dict[str, object]] = []
    for item in spectra:
        mask = (item.energy_mev >= SPECTRUM_RANGE[0]) & (item.energy_mev <= SPECTRUM_RANGE[1])
        for index in np.flatnonzero(mask):
            spectrum_rows.append(
                {
                    "family": item.metadata["family"],
                    "edf": item.metadata["edf"],
                    "boost_amplitude": item.boost_amplitude,
                    "propagation_time_fm_c": item.propagation_time_fm_c,
                    "rayleigh_resolution_mev": item.rayleigh_resolution_mev,
                    "zero_padded_energy_bin_mev": item.energy_bin_mev,
                    "channel": item.channel.name,
                    "gamma_mev": item.window.gamma_mev,
                    "energy_mev": float(item.energy_mev[index]),
                    "response_real": float(item.response.real[index]),
                    "response_imag": float(item.response.imag[index]),
                    "signed_strength": float(item.signed_strength[index]),
                }
            )
    write_csv(output / "spectra.csv", spectrum_rows)

    with h5py.File(output / "response_convergence.h5", "w", track_order=True) as handle:
        handle.attrs["schema"] = "sky3d-two-edf-highres-response-v2"
        handle.attrs["hbar_c_mev_fm"] = HBARC_MEV_FM
        handle.attrs["zero_padding_factor"] = ZERO_PADDING
        handle.attrs["zero_padding_interpretation"] = "display-grid interpolation only"
        for item in spectra:
            family = handle.require_group(str(item.metadata["family"]))
            duration = family.require_group(f"T{item.propagation_time_fm_c:g}")
            channel = duration.require_group(item.channel.name)
            group = channel.create_group(item.window.name)
            group.attrs["edf"] = str(item.metadata["edf"])
            group.attrs["boost_amplitude"] = item.boost_amplitude
            group.attrs["gamma_mev"] = float(item.window.gamma_mev)
            group.attrs["rayleigh_resolution_mev"] = item.rayleigh_resolution_mev
            group.attrs["zero_padded_energy_bin_mev"] = item.energy_bin_mev
            group.attrs["window_end_amplitude"] = float(item.window_values[-1])
            for name, values in (
                ("time_fm_c", item.time_fm_c),
                ("raw_signal", item.raw_signal),
                ("reference_signal", item.reference_signal),
                ("reference_delta", item.reference_delta),
                ("delta_signal", item.delta_signal),
                ("window_value", item.window_values),
                ("energy_mev", item.energy_mev),
                ("response_real", item.response.real),
                ("response_imag", item.response.imag),
                ("signed_strength", item.signed_strength),
            ):
                group.create_dataset(name, data=values, compression="gzip")


def plot_resolution(spectra: list[Spectrum], output: Path) -> None:
    lookup = {spectrum_key(item): item for item in spectra}
    families = (("sly5_n24_e0", "SLy5"), ("skms_n24_e0", "SkM*"))
    if any((family, 18000.0, "e0_diagonal", 0.1) not in lookup for family, _ in families):
        return
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.05,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.35, 4.15), sharex=True, constrained_layout=True)
    duration_colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(DURATIONS)))
    gamma_colors = plt.cm.plasma(np.linspace(0.05, 0.9, len(GAMMAS)))
    for row, (family, edf) in enumerate(families):
        for duration, color in zip(DURATIONS, duration_colors):
            item = lookup[(family, duration, "e0_diagonal", 0.2)]
            mask = (item.energy_mev >= 9.0) & (item.energy_mev <= 25.0)
            axes[row, 0].plot(
                item.energy_mev[mask], item.signed_strength[mask], color=color,
                label=f"{duration/1000:g}k",
            )
        for gamma, color in zip(GAMMAS, gamma_colors):
            item = lookup[(family, 18000.0, "e0_diagonal", gamma)]
            mask = (item.energy_mev >= 9.0) & (item.energy_mev <= 25.0)
            axes[row, 1].plot(
                item.energy_mev[mask], item.signed_strength[mask], color=color,
                label=f"{gamma:g}",
            )
        axes[row, 0].set_ylabel(f"{edf}\n" + r"$S_{E0}$ (fm$^4$/MeV)")
    axes[0, 0].set_title(r"$\Gamma_{\rm sm}=0.2$ MeV")
    axes[0, 1].set_title(r"$T=18000$ fm/$c$")
    axes[0, 0].legend(
        title=r"$T$ (10$^3$ fm/$c$)", title_fontsize=7.2, frameon=False, ncol=3
    )
    axes[0, 1].legend(
        title=r"$\Gamma_{\rm sm}$ (MeV)", title_fontsize=7.2, frameon=False, ncol=3
    )
    for axis in axes.flat:
        axis.set_xlim(9.0, 25.0)
        axis.tick_params(direction="in", top=True, right=True)
        axis.grid(color="0.9", linewidth=0.45)
    for axis in axes[-1, :]:
        axis.set_xlabel("Excitation energy (MeV)")
    for suffix in ("pdf", "png"):
        fig.savefig(
            output / f"highres_time_smoothing_two_edf.{suffix}",
            dpi=300,
        )
    plt.close(fig)


def main() -> None:
    arguments = parser().parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    spectra: list[Spectrum] = []
    density: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []

    families: list[tuple[str, str, Path, Path, str, tuple[float, ...], float]] = []
    required = (
        ("sly5_n24_e0", "SLy5", arguments.sly5_e0, arguments.sly5_reference, "e0", ETA),
        ("skms_n24_e0", "SkM*", arguments.skms_e0, arguments.skms_reference, "e0", ETA),
    )
    optional = (
        ("sly5_n24_e2", "SLy5", arguments.sly5_e2, arguments.sly5_reference, "e2", ETA),
        ("skms_n24_e2", "SkM*", arguments.skms_e2, arguments.skms_reference, "e2", ETA),
        ("sly5_n24_e0_half", "SLy5", arguments.sly5_half, arguments.sly5_reference, "e0", ETA / 2.0),
        ("skms_n24_e0_half", "SkM*", arguments.skms_half, arguments.skms_reference, "e0", ETA / 2.0),
        ("sly5_n32_e0", "SLy5", arguments.sly5_large_e0, arguments.sly5_large_reference, "e0", ETA),
        ("skms_n32_e0", "SkM*", arguments.skms_large_e0, arguments.skms_large_reference, "e0", ETA),
    )
    for family, edf, run, reference, mode, boost_amplitude in required + optional:
        if run is None:
            continue
        if reference is None:
            raise ValueError(f"{family}: boosted and reference directories must both be provided")
        final_time = numeric_table(run / "energies.res")[-1, 0]
        durations = tuple(duration for duration in DURATIONS if duration <= final_time + 1.0e-8)
        if not durations:
            raise ValueError(f"{family}: no requested duration is available")
        families.append((family, edf, run, reference, mode, durations, boost_amplitude))

    seen_density: set[tuple[Path, str]] = set()
    for family, edf, run, reference, mode, durations, boost_amplitude in families:
        run = run.resolve()
        reference = reference.resolve()
        spectra.extend(
            analyze_family(
                family=family,
                edf=edf,
                run=run,
                reference_run=reference,
                mode=mode,
                durations=durations,
                boost_amplitude=boost_amplitude,
            )
        )
        for role, directory in ((mode, run), ("reference", reference)):
            density_key = (directory, f"{family}:{role}")
            if density_key in seen_density:
                continue
            rows = density_rows(directory, family, role)
            density.extend(rows)
            stability.extend(stability_rows(directory, family, role, rows, durations))
            seen_density.add(density_key)

    summaries = summary_rows(spectra)
    convergence = convergence_rows(spectra)
    reciprocity = reciprocity_rows(spectra)
    linearity = linearity_rows(spectra)
    box_size = box_size_rows(spectra)
    write_csv(output / "summary.csv", summaries)
    write_csv(output / "stability.csv", stability)
    write_csv(output / "density_diagnostics.csv", density)
    write_csv(output / "time_convergence.csv", convergence)
    write_csv(output / "reciprocity.csv", reciprocity)
    write_csv(output / "linearity.csv", linearity)
    write_csv(output / "box_size.csv", box_size)
    write_spectrum_products(output, spectra)
    plot_resolution(spectra, FIGURES)

    rayleigh = {
        f"T{duration:g}": 2.0 * math.pi * HBARC_MEV_FM / duration
        for duration in DURATIONS
    }
    metadata = {
        "schema": "sky3d-two-edf-highres-response-v2",
        "durations_fm_c": DURATIONS,
        "gamma_sm_mev": GAMMAS,
        "rayleigh_resolution_mev": rayleigh,
        "zero_padding_factor": ZERO_PADDING,
        "window_end_acceptance_threshold": WINDOW_END_THRESHOLD,
        "zero_padding_statement": (
            "The zero-padded energy bin is a plotting/interpolation grid and is not spectral resolution."
        ),
        "exponential_window": "exp[-Gamma_sm t/(2 hbar)]; Gamma_sm is imposed Lorentzian FWHM",
        "families": [
            {
                "name": family,
                "edf": edf,
                "run": str(Path(run).resolve()),
                "reference": str(Path(reference).resolve()),
                "mode": mode,
                "durations": durations,
                "boost_amplitude": boost_amplitude,
            }
            for family, edf, run, reference, mode, durations, boost_amplitude in families
        ],
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata["rayleigh_resolution_mev"], indent=2, sort_keys=True))
    print(f"Wrote {len(spectra)} spectra and {len(summaries)} summary rows to {output}")


if __name__ == "__main__":
    main()
