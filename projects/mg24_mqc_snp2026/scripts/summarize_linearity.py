#!/usr/bin/env python3
"""Compare baseline and half-amplitude E0 response calculations."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "data" / "processed" / "e0_baseline"
HALF = PROJECT / "data" / "processed" / "e0_half"
RUN_BASE = PROJECT / "runs" / "e0_baseline"
RUN_HALF = PROJECT / "runs" / "e0_half"
RUN_REFERENCE = PROJECT / "runs" / "no_boost_long"
PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"
ETA_BASE = 5.0e-5
ETA_HALF = 2.5e-5


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def spectrum(data: list[dict[str, str]], channel: str, window: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in data if row["channel"] == channel and row["window"] == window]
    energy = np.asarray([float(row["energy_mev"]) for row in selected])
    response = np.asarray(
        [complex(float(row["response_real"]), float(row["response_imag"])) for row in selected]
    )
    return energy, response


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.linalg.norm(first - second) /
        (0.5 * (np.linalg.norm(first) + np.linalg.norm(second)))
    )


def metrics(base: np.ndarray, half: np.ndarray) -> dict[str, float]:
    scale = float(np.vdot(base, half).real / np.vdot(base, base).real)
    return {
        "symmetric_relative_l2": relative_l2(base, half),
        "best_real_scale_half_over_baseline": scale,
        "relative_l2_after_best_real_scale": float(
            np.linalg.norm(half - scale * base) / np.linalg.norm(half)
        ),
        "max_abs_difference": float(np.max(np.abs(half - base))),
    }


def protocol(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, comments="#"))


def main() -> None:
    base_rows = rows(BASE / "spectra.csv")
    half_rows = rows(HALF / "spectra.csv")
    result: dict[str, object] = {
        "eta_baseline": ETA_BASE,
        "eta_half": ETA_HALF,
        "amplitude_ratio": ETA_HALF / ETA_BASE,
        "comparison": "all time signals and spectra are divided by their own boost eta",
        "windows": {},
    }
    regions = {"mqc": (9.0, 18.0), "experiment_range": (9.0, 25.0)}
    for window in ("gamma_1", "gamma_2", "cosine_6"):
        window_result: dict[str, object] = {}
        for channel in ("e0_diagonal", "q20_from_e0"):
            energy_base, response_base = spectrum(base_rows, channel, window)
            energy_half, response_half = spectrum(half_rows, channel, window)
            if not np.array_equal(energy_base, energy_half):
                raise ValueError("energy grids differ")
            channel_result: dict[str, object] = {}
            for name, (low, high) in regions.items():
                mask = (energy_base >= low) & (energy_base <= high)
                complex_result = metrics(response_base[mask], response_half[mask])
                signed_result = metrics(
                    response_base[mask].imag / np.pi, response_half[mask].imag / np.pi
                )
                signed_result["pearson_correlation"] = float(
                    np.corrcoef(response_base[mask].imag, response_half[mask].imag)[0, 1]
                )
                channel_result[name] = {
                    "complex_response": complex_result,
                    "signed_strength": signed_result,
                }
            window_result[channel] = channel_result
        result["windows"][window] = window_result

    mono_base = protocol(RUN_BASE / "monopoles.res")
    mono_half = protocol(RUN_HALF / "monopoles.res")
    quad_base = protocol(RUN_BASE / "quadrupoles.res")
    quad_half = protocol(RUN_HALF / "quadrupoles.res")
    mono_reference = protocol(RUN_REFERENCE / "monopoles.res")
    quad_reference = protocol(RUN_REFERENCE / "quadrupoles.res")
    if not (
        np.array_equal(mono_base[:, 0], mono_half[:, 0])
        and np.array_equal(quad_base[:, 0], quad_half[:, 0])
        and np.array_equal(mono_base[:, 0], mono_reference[:, 0])
        and np.array_equal(quad_base[:, 0], quad_reference[:, 0])
    ):
        raise ValueError("time grids differ")
    mono_background = mono_reference[:, 1] - mono_reference[0, 1]
    quad_background = quad_reference[:, 1] - quad_reference[0, 1]
    mono_delta_base = mono_base[:, 1] - mono_base[0, 1]
    mono_delta_half = mono_half[:, 1] - mono_half[0, 1]
    quad_delta_base = quad_base[:, 1] - quad_base[0, 1]
    quad_delta_half = quad_half[:, 1] - quad_half[0, 1]
    time_mono_base = (mono_delta_base - mono_background) / ETA_BASE
    time_mono_half = (mono_delta_half - mono_background) / ETA_HALF
    time_quad_base = (quad_delta_base - quad_background) / ETA_BASE
    time_quad_half = (quad_delta_half - quad_background) / ETA_HALF
    result["time_domain_delta_q_over_eta"] = {
        "q00": metrics(time_mono_base[1:], time_mono_half[1:]),
        "q20": metrics(time_quad_base[1:], time_quad_half[1:]),
    }
    result["matched_no_boost_reference"] = {
        "definition": "corrected DeltaQ_eta(t) = boosted DeltaQ_eta(t) - no-boost DeltaQ_0(t)",
        "max_abs_q00_background": float(np.max(np.abs(mono_background))),
        "max_abs_q20_background": float(np.max(np.abs(quad_background))),
        "max_abs_q00_nonlinear_residual": float(
            np.max(np.abs(2.0 * (mono_delta_half - mono_background) - (mono_delta_base - mono_background)))
        ),
        "max_abs_q20_nonlinear_residual": float(
            np.max(np.abs(2.0 * (quad_delta_half - quad_background) - (quad_delta_base - quad_background)))
        ),
    }

    (PROCESSED / "linearity_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 220})
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.8), sharex=True, constrained_layout=True)
    for axis, channel, ylabel in (
        (axes[0], "e0_diagonal", r"$S_{00}(E)$ (fm$^4$/MeV)"),
        (axes[1], "q20_from_e0", r"signed $S_{20,00}(E)$ (fm$^4$/MeV)"),
    ):
        energy, response_base = spectrum(base_rows, channel, "gamma_1")
        _, response_half = spectrum(half_rows, channel, "gamma_1")
        mask = (energy >= 9.0) & (energy <= 25.0)
        axis.plot(energy[mask], response_base[mask].imag / np.pi, color="#1f4e79", lw=1.35, label=r"$\eta=5\times10^{-5}$")
        axis.plot(energy[mask], response_half[mask].imag / np.pi, color="#b44726", lw=1.15, ls="--", label=r"$\eta=2.5\times10^{-5}$")
        axis.axhline(0.0, color="0.3", lw=0.65)
        axis.axvspan(10.0, 18.0, color="#e7a33e", alpha=0.10, lw=0)
        axis.axvspan(18.0, 25.0, color="#4d83b3", alpha=0.07, lw=0)
        axis.set_ylabel(ylabel)
        axis.set_xlim(9.0, 25.0)
        axis.grid(color="0.88", linewidth=0.55)
        axis.tick_params(direction="in", top=True, right=True)
        axis.legend(frameon=False)
    axes[0].set_title(r"$^{24}$Mg E0 boost linearity ($\Delta Q/\eta$, $\Gamma_{\rm sm}=1$ MeV)")
    axes[1].set_xlabel("Excitation energy (MeV)")
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"e0_linearity.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
