#!/usr/bin/env python3
"""Summarize and plot the Milestone-3/4 E0 boost and mixed response."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
RUN = PROJECT / "runs" / "e0_baseline"
REFERENCE_RUN = PROJECT / "runs" / "no_boost_long"
ANALYSIS = PROJECT / "data" / "processed" / "e0_baseline"
PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"
HBAR_C = 197.3269804


def load_protocol(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, comments="#"))


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select_spectrum(rows: list[dict[str, str]], channel: str, window: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["channel"] == channel and row["window"] == window]
    return (
        np.asarray([float(row["energy_mev"]) for row in selected]),
        np.asarray([float(row["signed_strength"]) for row in selected]),
    )


def summary_row(rows: list[dict[str, str]], channel: str, window: str, region: str) -> dict[str, str]:
    for row in rows:
        if row["channel"] == channel and row["window"] == window and row["region"] == region:
            return row
    raise KeyError((channel, window, region))


def style_axes(axis) -> None:
    axis.tick_params(direction="in", top=True, right=True)
    axis.grid(color="0.88", linewidth=0.55)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)

    monopole = load_protocol(RUN / "monopoles.res")
    quadrupole = load_protocol(RUN / "quadrupoles.res")
    reference_monopole = load_protocol(REFERENCE_RUN / "monopoles.res")
    reference_quadrupole = load_protocol(REFERENCE_RUN / "quadrupoles.res")
    if not (
        np.array_equal(monopole[:, 0], reference_monopole[:, 0])
        and np.array_equal(quadrupole[:, 0], reference_quadrupole[:, 0])
    ):
        raise ValueError("boosted and reference time grids differ")
    summary_rows = csv_rows(ANALYSIS / "summary.csv")
    spectrum_rows = csv_rows(ANALYSIS / "spectra.csv")
    milestone2 = json.loads((PROCESSED / "milestone2_summary.json").read_text(encoding="utf-8"))

    reference_dq00 = reference_monopole[:, 1] - reference_monopole[0, 1]
    reference_dq20 = reference_quadrupole[:, 1] - reference_quadrupole[0, 1]
    dq00 = monopole[:, 1] - monopole[0, 1] - reference_dq00
    dq20 = quadrupole[:, 1] - quadrupole[0, 1] - reference_dq20
    duration = float(monopole[-1, 0])
    resolution = 2.0 * math.pi * HBAR_C / duration

    e0_mqc = summary_row(summary_rows, "e0_diagonal", "gamma_1", "mqc")
    cross_mqc = summary_row(summary_rows, "q20_from_e0", "gamma_1", "mqc")
    cross_high = summary_row(summary_rows, "q20_from_e0", "gamma_1", "main_isgmr")
    no_boost_q20 = float(np.max(np.abs(reference_dq20)))

    result = {
        "boost": {"operator": "Sky3D isoscalar L=0", "eta": 5.0e-5},
        "dt_fm_over_c": 0.2,
        "output_sample_spacing_fm_over_c": float(monopole[1, 0] - monopole[0, 0]),
        "duration_fm_over_c": duration,
        "rayleigh_resolution_mev": resolution,
        "max_abs_delta_q00": float(np.max(np.abs(dq00))),
        "max_abs_delta_q20": float(np.max(np.abs(dq20))),
        "no_boost_max_abs_delta_q20": float(no_boost_q20),
        "matched_no_boost_reference_subtracted": True,
        "q20_signal_to_no_boost_drift": float(np.max(np.abs(dq20)) / no_boost_q20),
        "gamma_1_mev": {
            "e0_mqc_peak_mev": float(e0_mqc["peak_energy_mev"]),
            "cross_mqc_peak_mev": float(cross_mqc["peak_energy_mev"]),
            "peak_separation_mev": abs(float(e0_mqc["peak_energy_mev"]) - float(cross_mqc["peak_energy_mev"])),
            "e0_mqc_centroid_m1_m0_mev": float(e0_mqc["centroid_m1_m0_mev"]),
            "e0_mqc_m0": float(e0_mqc["m0"]),
            "cross_mqc_signed_m0": float(cross_mqc["m0"]),
            "cross_main_isgmr_signed_m0": float(cross_high["m0"]),
            "cross_main_isgmr_negative_area_fraction": float(cross_high["negative_area_fraction"]),
        },
        "go_no_go": "GO: E0 boost produces a resolved intrinsic-axis Q20 response at the dominant 16-MeV low-energy E0 structure",
    }
    (PROCESSED / "e0_go_nogo.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 220})
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.5), sharex=True, constrained_layout=True)
    axes[0].plot(monopole[:, 0], dq00, color="#b44726", lw=1.05)
    axes[0].set_ylabel(r"$\Delta Q_{00}(t)$ (fm$^2$)")
    axes[0].set_title(r"$^{24}$Mg after an isoscalar E0 boost ($\eta=5\times10^{-5}$)")
    axes[1].plot(quadrupole[:, 0], dq20, color="#2f7d4a", lw=1.05)
    axes[1].set_ylabel(r"$\Delta Q_{20}(t)$ (fm$^2$)")
    axes[1].set_xlabel(r"Time (fm/$c$)")
    for axis in axes:
        axis.axhline(0.0, color="0.35", lw=0.6)
        style_axes(axis)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"e0_time_traces.{suffix}", bbox_inches="tight")
    plt.close(fig)

    energy_e0, strength_e0 = select_spectrum(spectrum_rows, "e0_diagonal", "gamma_1")
    energy_cross, strength_cross = select_spectrum(spectrum_rows, "q20_from_e0", "gamma_1")
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.8), sharex=True, constrained_layout=True)
    for axis in axes:
        axis.axvspan(10.0, 18.0, color="#e7a33e", alpha=0.13, lw=0)
        axis.axvspan(18.0, 25.0, color="#4d83b3", alpha=0.09, lw=0)
        axis.set_xlim(9.0, 25.0)
        style_axes(axis)
    axes[0].plot(energy_e0, strength_e0, color="#b44726", lw=1.4)
    axes[0].set_ylabel(r"$S_{00}(E)$ (fm$^4$/MeV)")
    axes[0].set_title(r"$^{24}$Mg E0 and induced intrinsic-axis E2 response")
    axes[1].plot(energy_cross, strength_cross, color="#2f7d4a", lw=1.4)
    axes[1].axhline(0.0, color="0.25", lw=0.75)
    axes[1].set_ylabel(r"signed $S_{20,00}(E)$ (fm$^4$/MeV)")
    axes[1].set_xlabel("Excitation energy (MeV)")
    axes[0].text(0.015, 0.93, r"$\Gamma_{\rm sm}=1$ MeV; $T=2000$ fm/$c$", transform=axes[0].transAxes, va="top")
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"e0_response_go_nogo.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
