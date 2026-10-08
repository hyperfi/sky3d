#!/usr/bin/env python3
"""Generate page-safe SLy5/SkM* figures from the two-EDF response products."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
EXPERIMENTAL = PROJECT / "data" / "experimental"
FIGURES = PROJECT / "figures"
FOUR_PI = 4.0 * math.pi
ENERGY_LIMITS = (9.0, 25.0)
PAGE_WIDTH_IN = 6.35
COLUMN_FIGURE_WIDTH_IN = 3.25


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--spectra",
        type=Path,
        default=PROCESSED / "highres" / "spectra.csv",
    )
    result.add_argument(
        "--experiment",
        type=Path,
        default=EXPERIMENTAL / "digitized_is0_bahini2022.csv",
    )
    result.add_argument("--gamma", type=float, default=0.2)
    result.add_argument(
        "--experiment-gamma",
        type=float,
        default=1.0,
        help="Artificial FWHM for the experiment/theory overview only.",
    )
    result.add_argument("--duration", type=float, default=18000.0)
    result.add_argument("--output", type=Path, default=FIGURES)
    result.add_argument(
        "--summary-output",
        type=Path,
        default=PROCESSED / "experimental_comparison_two_edf.json",
    )
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_spectrum(
    data: list[dict[str, str]],
    *,
    family: str,
    channel: str,
    duration: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in data
        if row["family"] == family
        and row["channel"] == channel
        and math.isclose(float(row["propagation_time_fm_c"]), duration)
        and math.isclose(float(row["gamma_mev"]), gamma)
    ]
    if not selected:
        raise ValueError(
            f"missing spectrum: family={family}, channel={channel}, "
            f"T={duration:g} fm/c, Gamma={gamma:g} MeV"
        )
    selected.sort(key=lambda row: float(row["energy_mev"]))
    return (
        np.asarray([float(row["energy_mev"]) for row in selected]),
        np.asarray([float(row["signed_strength"]) for row in selected]),
    )


def experimental_arrays(
    data: list[dict[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray([float(row["energy_mev"]) for row in data]),
        np.asarray([float(row["is0_strength_fm4_per_mev"]) for row in data]),
        np.asarray(
            [float(row["experimental_uncertainty_minus_fm4_per_mev"]) for row in data]
        ),
        np.asarray(
            [float(row["experimental_uncertainty_plus_fm4_per_mev"]) for row in data]
        ),
    )


def bounded(energy: np.ndarray, strength: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (energy >= ENERGY_LIMITS[0]) & (energy <= ENERGY_LIMITS[1])
    return energy[mask], strength[mask]


def moment_summary(
    energy: np.ndarray, strength: np.ndarray, low: float, high: float
) -> dict[str, float]:
    mask = (energy >= low) & (energy <= high)
    selected_energy = energy[mask]
    selected_strength = strength[mask]
    m0 = float(np.trapz(selected_strength, selected_energy))
    m1 = float(np.trapz(selected_energy * selected_strength, selected_energy))
    peak = float(selected_energy[int(np.argmax(selected_strength))])
    return {
        "m0_fm4": m0,
        "m1_mev_fm4": m1,
        "centroid_mev": m1 / m0 if m0 > 0.0 else float("nan"),
        "peak_energy_mev": peak,
    }


def experimental_moment_summary(
    energy: np.ndarray, strength: np.ndarray, low: float, high: float
) -> dict[str, float]:
    mask = (energy >= low) & (energy <= high)
    selected_energy = energy[mask]
    selected_strength = strength[mask]
    bin_width = float(np.median(np.diff(energy)))
    m0 = float(np.sum(selected_strength) * bin_width)
    m1 = float(np.sum(selected_energy * selected_strength) * bin_width)
    return {
        "m0_fm4": m0,
        "m1_mev_fm4": m1,
        "centroid_mev": m1 / m0 if m0 > 0.0 else float("nan"),
        "maximum_bin_energy_mev": float(selected_energy[int(np.argmax(selected_strength))]),
    }


def style(axis: plt.Axes, *, zero_line: bool = False) -> None:
    axis.set_xlim(*ENERGY_LIMITS)
    axis.tick_params(direction="in", top=True, right=True)
    axis.grid(color="0.91", linewidth=0.45)
    axis.axvspan(9.0, 18.0, color="0.75", alpha=0.10, linewidth=0)
    if zero_line:
        axis.axhline(0.0, color="0.35", linewidth=0.6)


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    arguments = parser().parse_args()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spectra = read_rows(arguments.spectra)
    experiment = read_rows(arguments.experiment)

    selections: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for family in ("sly5_n24_e0", "skms_n24_e0"):
        for channel in ("e0_diagonal", "q20_from_e0"):
            selections[f"{family}:{channel}"] = select_spectrum(
                spectra,
                family=family,
                channel=channel,
                duration=arguments.duration,
                gamma=arguments.gamma,
            )
    for channel in ("e2k0_diagonal", "q00_from_e2k0"):
        selections[f"sly5_n24_e2:{channel}"] = select_spectrum(
            spectra,
            family="sly5_n24_e2",
            channel=channel,
            duration=arguments.duration,
            gamma=arguments.gamma,
        )

    e_exp, s_exp, err_minus, err_plus = experimental_arrays(experiment)
    e_sly, s_sly_code = select_spectrum(
        spectra,
        family="sly5_n24_e0",
        channel="e0_diagonal",
        duration=arguments.duration,
        gamma=arguments.experiment_gamma,
    )
    e_skms, s_skms_code = select_spectrum(
        spectra,
        family="skms_n24_e0",
        channel="e0_diagonal",
        duration=arguments.duration,
        gamma=arguments.experiment_gamma,
    )
    e_sly, s_sly = bounded(e_sly, FOUR_PI * s_sly_code)
    e_skms, s_skms = bounded(e_skms, FOUR_PI * s_skms_code)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
        }
    )

    # Figure 1: experiment and the two EDFs with identical theoretical smoothing.
    fig, axis = plt.subplots(
        figsize=(COLUMN_FIGURE_WIDTH_IN, 1.75), constrained_layout=True
    )
    axis.errorbar(
        e_exp,
        s_exp,
        yerr=np.vstack((err_minus, err_plus)),
        fmt="o",
        ms=3.2,
        mew=0.6,
        color="0.10",
        ecolor="0.42",
        elinewidth=0.8,
        capsize=1.5,
        label="Experiment",
        zorder=4,
    )
    axis.plot(e_sly, s_sly, color="#b44726", linewidth=1.45, label="SLy5")
    axis.plot(
        e_skms,
        s_skms,
        color="#315f9b",
        linewidth=1.45,
        linestyle="--",
        label="SkM*",
    )
    axis.set_xlabel("Excitation energy (MeV)")
    axis.set_ylabel("IS0 strength")
    axis.set_ylim(bottom=0.0)
    axis.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        handlelength=1.8,
        labelspacing=0.15,
    )
    style(axis)
    save(fig, output, "fig1_e0_experiment_two_edf")

    e_e0, s_e0 = bounded(*selections["sly5_n24_e0:e0_diagonal"])
    e_e2, s_e2 = bounded(*selections["sly5_n24_e2:e2k0_diagonal"])
    e_cross, s_cross = bounded(*selections["sly5_n24_e0:q20_from_e0"])
    e_reverse, s_reverse = bounded(*selections["sly5_n24_e2:q00_from_e2k0"])
    e_cross_skms, s_cross_skms = bounded(*selections["skms_n24_e0:q20_from_e0"])

    # Figure 2: SLy5 diagonal and signed mixed responses on aligned energy axes.
    fig, axes = plt.subplots(
        3, 1, figsize=(PAGE_WIDTH_IN, 5.15), sharex=True, constrained_layout=True
    )
    axes[0].plot(e_e0, s_e0, color="#b44726")
    axes[0].set_ylabel(r"$S_{00}$")
    axes[0].text(0.015, 0.88, "(a)", transform=axes[0].transAxes)
    axes[1].plot(e_e2, s_e2, color="#315f9b")
    axes[1].set_ylabel(r"$S_{20}$")
    axes[1].text(0.015, 0.88, "(b)", transform=axes[1].transAxes)
    axes[2].plot(e_cross, s_cross, color="#2f7d4a", label=r"$Q_{20}\leftarrow E0$")
    axes[2].plot(
        e_reverse,
        s_reverse,
        color="#6e4b8b",
        linestyle="--",
        label=r"$Q_{00}\leftarrow E2$",
    )
    axes[2].set_ylabel(r"$R_{20,00}$")
    axes[2].set_xlabel("Excitation energy (MeV)")
    axes[2].legend(frameon=False, ncol=2, loc="upper right")
    for axis in axes:
        style(axis, zero_line=True)
    save(fig, output, "fig2_sly5_diagonal_cross")

    # Figure 3: the direct EDF comparison of the forward signed cross response.
    fig, axis = plt.subplots(figsize=(PAGE_WIDTH_IN, 2.65), constrained_layout=True)
    axis.plot(e_cross, s_cross, color="#b44726", label="SLy5")
    axis.plot(e_cross_skms, s_cross_skms, color="#315f9b", linestyle="--", label="SkM*")
    axis.set_xlabel("Excitation energy (MeV)")
    axis.set_ylabel(r"$R_{20,00}$ (fm$^4$ MeV$^{-1}$)")
    axis.legend(frameon=False, ncol=2, loc="upper right")
    style(axis, zero_line=True)
    save(fig, output, "fig3_cross_response_two_edf")

    # Compact two-page option: the full physics comparison in one four-panel figure.
    fig, axes = plt.subplots(
        2, 2, figsize=(PAGE_WIDTH_IN, 4.55), sharex=True, constrained_layout=True
    )
    axes[0, 0].plot(e_e0, s_e0, color="#b44726")
    axes[0, 0].set_ylabel(r"$S_{00}$")
    axes[0, 0].text(0.025, 0.87, "(a)", transform=axes[0, 0].transAxes)
    axes[0, 1].plot(e_e2, s_e2, color="#315f9b")
    axes[0, 1].set_ylabel(r"$S_{20}$")
    axes[0, 1].text(0.025, 0.87, "(b)", transform=axes[0, 1].transAxes)
    axes[1, 0].plot(e_cross, s_cross, color="#2f7d4a", label=r"$R_{20,00}$")
    axes[1, 0].plot(
        e_reverse,
        s_reverse,
        color="#6e4b8b",
        linestyle="--",
        label=r"$R_{00,20}$",
    )
    axes[1, 0].set_ylabel(r"mixed response")
    axes[1, 0].text(0.025, 0.87, "(c)", transform=axes[1, 0].transAxes)
    axes[1, 0].legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        handlelength=1.8,
        labelspacing=0.15,
    )
    axes[1, 1].plot(e_cross, s_cross, color="#b44726", label="SLy5")
    axes[1, 1].plot(
        e_cross_skms, s_cross_skms, color="#315f9b", linestyle="--", label="SkM*"
    )
    axes[1, 1].set_ylabel(r"$R_{20,00}$")
    axes[1, 1].text(0.025, 0.87, "(d)", transform=axes[1, 1].transAxes)
    axes[1, 1].legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        handlelength=1.8,
        labelspacing=0.15,
    )
    for axis in axes.flat:
        style(axis, zero_line=True)
    for axis in axes[1, :]:
        axis.set_xlabel("Excitation energy (MeV)")
    save(fig, output, "fig2_combined_mqc_two_edf")

    summary = {
        "duration_fm_c": arguments.duration,
        "rayleigh_resolution_mev": 2.0 * math.pi * 197.3269804 / arguments.duration,
        "experiment_figure_artificial_smoothing_fwhm_mev": arguments.experiment_gamma,
        "response_figure_artificial_smoothing_fwhm_mev": arguments.gamma,
        "theory_convention": (
            "Sky3D F00 strength multiplied by 4*pi to match experimental sum_i r_i^2; "
            "no fitted normalization or energy shift."
        ),
        "experiment": {
            "mqc_10_18": experimental_moment_summary(e_exp, s_exp, 10.0, 18.0),
            "main_18_25": experimental_moment_summary(e_exp, s_exp, 18.0, 25.0),
        },
        "SLy5": {
            "mqc_10_18": moment_summary(e_sly, s_sly, 10.0, 18.0),
            "main_18_25": moment_summary(e_sly, s_sly, 18.0, 25.0),
        },
        "SkM*": {
            "mqc_10_18": moment_summary(e_skms, s_skms, 10.0, 18.0),
            "main_18_25": moment_summary(e_skms, s_skms, 18.0, 25.0),
        },
    }
    arguments.summary_output.parent.mkdir(parents=True, exist_ok=True)
    arguments.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
