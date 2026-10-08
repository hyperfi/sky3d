#!/usr/bin/env python3
"""Compare the independently calculated E0/E2 off-diagonal responses."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
E0 = PROJECT / "data" / "processed" / "e0_baseline"
E2 = PROJECT / "data" / "processed" / "e2k0_baseline"
RUN_E0 = PROJECT / "runs" / "e0_baseline"
RUN_E2 = PROJECT / "runs" / "e2k0_baseline"
RUN_REFERENCE = PROJECT / "runs" / "no_boost_long"
PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"
ETA = 5.0e-5


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def spectrum(rows: list[dict[str, str]], channel: str, window: str) -> tuple[np.ndarray, np.ndarray]:
    selected = [row for row in rows if row["channel"] == channel and row["window"] == window]
    energy = np.asarray([float(row["energy_mev"]) for row in selected])
    response = np.asarray(
        [complex(float(row["response_real"]), float(row["response_imag"])) for row in selected]
    )
    return energy, response


def relative_l2(first: np.ndarray, second: np.ndarray) -> float:
    denominator = 0.5 * (np.linalg.norm(first) + np.linalg.norm(second))
    return float(np.linalg.norm(first - second) / denominator)


def compare(first: np.ndarray, second: np.ndarray) -> dict[str, float]:
    denom = float(np.vdot(first, first).real)
    scale = float(np.vdot(first, second).real / denom)
    return {
        "symmetric_relative_l2": relative_l2(first, second),
        "best_real_scale_second_over_first": scale,
        "relative_l2_after_best_real_scale": float(
            np.linalg.norm(second - scale * first) / np.linalg.norm(second)
        ),
        "max_abs_difference": float(np.max(np.abs(first - second))),
    }


def protocol(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, comments="#"))


def main() -> None:
    e0_rows = csv_rows(E0 / "spectra.csv")
    e2_rows = csv_rows(E2 / "spectra.csv")
    regions = {"mqc": (9.0, 18.0), "experiment_range": (9.0, 25.0)}
    result: dict[str, object] = {
        "expected_relation": (
            "chi_20,00(E) = chi_00,20(E) for two time-even Hermitian operators "
            "in a time-reversal-invariant reference, using identical operator and boost normalizations"
        ),
        "boost_eta_both_runs": ETA,
        "windows": {},
    }

    for window in ("gamma_1", "gamma_2", "cosine_6"):
        energy_a, response_a = spectrum(e0_rows, "q20_from_e0", window)
        energy_b, response_b = spectrum(e2_rows, "q00_from_e2k0", window)
        if not np.array_equal(energy_a, energy_b):
            raise ValueError("response energy grids differ")
        window_result: dict[str, object] = {}
        for name, (low, high) in regions.items():
            mask = (energy_a >= low) & (energy_a <= high)
            complex_metrics = compare(response_a[mask], response_b[mask])
            strength_a = response_a[mask].imag / np.pi
            strength_b = response_b[mask].imag / np.pi
            strength_metrics = compare(strength_a, strength_b)
            correlation = float(np.corrcoef(strength_a, strength_b)[0, 1])
            window_result[name] = {
                "complex_response": complex_metrics,
                "signed_strength": {**strength_metrics, "pearson_correlation": correlation},
            }
        result["windows"][window] = window_result

    q20_after_e0 = protocol(RUN_E0 / "quadrupoles.res")
    q00_after_e2 = protocol(RUN_E2 / "monopoles.res")
    q20_reference = protocol(RUN_REFERENCE / "quadrupoles.res")
    q00_reference = protocol(RUN_REFERENCE / "monopoles.res")
    if not (
        np.array_equal(q20_after_e0[:, 0], q00_after_e2[:, 0])
        and np.array_equal(q20_after_e0[:, 0], q20_reference[:, 0])
        and np.array_equal(q00_after_e2[:, 0], q00_reference[:, 0])
    ):
        raise ValueError("time grids differ")
    q20_background = q20_reference[:, 1] - q20_reference[0, 1]
    q00_background = q00_reference[:, 1] - q00_reference[0, 1]
    response_time_a = (
        q20_after_e0[:, 1] - q20_after_e0[0, 1] - q20_background
    ) / ETA
    response_time_b = (
        q00_after_e2[:, 1] - q00_after_e2[0, 1] - q00_background
    ) / ETA
    result["time_domain"] = compare(response_time_a[1:], response_time_b[1:])
    result["time_domain"]["matched_no_boost_reference"] = (
        "Each boosted DeltaQ(t) is corrected by the corresponding no-boost DeltaQ_0(t) "
        "before division by eta."
    )

    (PROCESSED / "reciprocity_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for window, window_result in result["windows"].items():
        for region, region_result in window_result.items():
            for component, metrics in region_result.items():
                rows.append({"window": window, "region": region, "component": component, **metrics})
    fieldnames = sorted({key for row in rows for key in row})
    with (PROCESSED / "reciprocity_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    energy, response_a = spectrum(e0_rows, "q20_from_e0", "gamma_1")
    _, response_b = spectrum(e2_rows, "q00_from_e2k0", "gamma_1")
    mask = (energy >= 9.0) & (energy <= 25.0)
    energy = energy[mask]
    response_a = response_a[mask]
    response_b = response_b[mask]

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 220})
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 7.0), sharex=True, constrained_layout=True)
    labels = (r"Im $\chi/\pi$ (fm$^4$/MeV)", r"Re $\chi$ (fm$^4$/MeV)")
    series = ((response_a.imag / np.pi, response_b.imag / np.pi), (response_a.real, response_b.real))
    for axis, ylabel, values in zip(axes[:2], labels, series):
        axis.plot(energy, values[0], color="#2f7d4a", lw=1.35, label=r"$Q_{20}\leftarrow E0$")
        axis.plot(energy, values[1], color="#6e4b8b", lw=1.15, ls="--", label=r"$Q_{00}\leftarrow E2(K=0)$")
        axis.axhline(0.0, color="0.3", lw=0.65)
        axis.set_ylabel(ylabel)
        axis.legend(frameon=False, loc="best")
    axes[0].set_title(r"$^{24}$Mg off-diagonal response reciprocity ($\Gamma_{\rm sm}=1$ MeV)")
    axes[2].plot(energy, (response_b.imag - response_a.imag) / np.pi, color="#b44726", lw=1.25)
    axes[2].axhline(0.0, color="0.3", lw=0.65)
    axes[2].set_ylabel(r"signed-strength residual")
    axes[2].set_xlabel("Excitation energy (MeV)")
    for axis in axes:
        axis.axvspan(10.0, 18.0, color="#e7a33e", alpha=0.10, lw=0)
        axis.axvspan(18.0, 25.0, color="#4d83b3", alpha=0.07, lw=0)
        axis.set_xlim(9.0, 25.0)
        axis.grid(color="0.88", linewidth=0.55)
        axis.tick_params(direction="in", top=True, right=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"cross_response_reciprocity.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
