#!/usr/bin/env python3
"""Combine K-resolved products, quantify linearity, and make final figures."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_WINDOW = "gamma_0p5"
CHANNELS = {
    "0": (ROOT / "response/K0/spectra.csv", "K0_code_native"),
    "1": (ROOT / "response/K1/spectra.csv", "K1_unit_tesseral"),
    "2": (ROOT / "response/K2/spectra.csv", "K2_unit_tesseral"),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def spectrum(path: Path, channel: str, window: str = PRIMARY_WINDOW):
    selected = [r for r in rows(path) if r["channel"] == channel and r["window"] == window]
    if not selected:
        raise ValueError(f"missing {channel}/{window} in {path}")
    energy = np.array([float(r["energy_mev"]) for r in selected])
    strength = np.array([float(r["signed_strength"]) for r in selected])
    keep = (energy >= 0.0) & (energy <= 35.0)
    return energy[keep], strength[keep]


def local_peaks(energy: np.ndarray, strength: np.ndarray) -> list[dict[str, float]]:
    region = (energy >= 10.0) & (energy <= 35.0)
    e, s = energy[region], strength[region]
    threshold = 0.10 * float(np.max(s))
    candidates = np.flatnonzero((s[1:-1] > s[:-2]) & (s[1:-1] >= s[2:])) + 1
    candidates = [i for i in candidates if s[i] >= threshold]
    candidates.sort(key=lambda i: s[i], reverse=True)
    accepted: list[int] = []
    for index in candidates:
        if all(abs(e[index] - e[old]) >= 0.5 for old in accepted):
            accepted.append(index)
        if len(accepted) == 4:
            break
    return [
        {"energy_mev": float(e[i]), "strength": float(s[i]), "relative_height": float(s[i] / np.max(s))}
        for i in accepted
    ]


def summary_row(component: str) -> dict[str, str]:
    path = ROOT / f"response/K{component}/summary.csv"
    channel = "K0_code_native" if component == "0" else f"K{component}_unit_tesseral"
    matches = [
        r for r in rows(path)
        if r["channel"] == channel and r["window"] == PRIMARY_WINDOW
        and r["region"] == "giant_quadrupole"
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one GQR summary row for K={component}")
    return matches[0]


def static_properties() -> dict[str, object]:
    text = (ROOT / "logs/static.log").read_text(encoding="utf-8", errors="replace")
    energy = float(re.findall(r"Total energy:\s+([-+0-9.Ee]+) MeV", text)[-1])
    total_rows = re.findall(
        r"Total:\s+([0-9.]+)\s+([0-9.]+)\s+([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)",
        text,
    )
    particle_number, rms, q20, x2, y2, z2 = map(float, total_rows[-1])
    beta_matches = re.findall(r"Beta:\s+([-+0-9.Ee]+)", text)
    gamma_matches = re.findall(r"Gamma:\s+([-+0-9.Ee]+)", text)
    beta = float(beta_matches[-1]) if beta_matches else q20 * (4.0 * np.pi) / (5.0 * (1.2 * particle_number ** (1.0 / 3.0)) ** 2 * particle_number)
    gamma = float(gamma_matches[-1]) if gamma_matches else 0.0
    rotation = json.loads((ROOT / "summary/axis_rotation.json").read_text(encoding="utf-8"))
    return {
        "energy_mev": energy,
        "particle_number": particle_number,
        "rms_radius_fm": rms,
        "q20_principal_fm2": q20,
        "beta": beta,
        "gamma_deg": gamma,
        "unrotated_second_moments_fm2": {"x": x2, "y": y2, "z": z2},
        "shape": "prolate" if beta > 0 and abs(gamma) < 1.0 else "not established as axial prolate",
        "production_axis": "z",
        "rotation": rotation,
    }


def normalized_time_signal(run: Path, eta: float, reference: Path):
    data = np.atleast_2d(np.loadtxt(run / "quadrupoles.res", comments="#"))
    ref = np.atleast_2d(np.loadtxt(reference / "quadrupoles.res", comments="#"))
    if data.shape != ref.shape or not np.array_equal(data[:, 0], ref[:, 0]):
        raise ValueError("K0 and reference time grids differ")
    corrected = (data[:, 1] - data[0, 1]) - (ref[:, 1] - ref[0, 1])
    return data[:, 0], corrected / eta


def main() -> None:
    output_rows: list[dict[str, object]] = []
    spectra: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for component, (path, channel) in CHANNELS.items():
        energy, strength = spectrum(path, channel)
        spectra[component] = (energy, strength)
        summary = summary_row(component)
        peaks = local_peaks(energy, strength)
        output_rows.append(
            {
                "K": int(component),
                "operator_normalization": "code native" if component == "0" else "unit real tesseral (2x code-native strength)",
                "dominant_peak_mev": peaks[0]["energy_mev"],
                "secondary_peaks_mev": ";".join(f'{p["energy_mev"]:.4f}' for p in peaks[1:]),
                "centroid_10_35_mev": float(summary["centroid_m1_m0_mev"]),
                "m0_10_35_fm4": float(summary["m0"]),
                "m1_10_35_mev_fm4": float(summary["m1"]),
                "negative_area_fraction_10_35": float(summary["negative_area_fraction"]),
                "rayleigh_resolution_mev": float(2.0 * np.pi * 197.3269804 / 6000.0),
                "artificial_smoothing_fwhm_mev": 0.5,
            }
        )

    e0, s0 = spectra["0"]
    eh, sh = spectrum(ROOT / "response/K0_half/spectra.csv", "K0_code_native")
    if not np.array_equal(e0, eh):
        raise ValueError("K0 baseline and half-strength energy grids differ")
    mask = (e0 >= 0.5) & (e0 <= 35.0)
    diff = s0[mask] - sh[mask]
    linearity = {
        "energy_interval_mev": [0.5, 35.0],
        "relative_l2": float(np.linalg.norm(diff) / np.linalg.norm(s0[mask])),
        "max_abs_difference_over_peak": float(np.max(np.abs(diff)) / np.max(s0[mask])),
        "eta_baseline": 5.0e-5,
        "eta_half": 2.5e-5,
    }

    static = static_properties()
    payload = {
        "schema": "icnpa2026-20ne-kresolved-e2-v1",
        "static": static,
        "components": output_rows,
        "linearity": linearity,
        "operator_note": (
            "K=0 is code native. For K=1,2, each positive-M code operator is Re(Y_lm) "
            "without sqrt(2); reported comparative strengths multiply the diagonal code response by 2."
        ),
    }
    (ROOT / "summary/results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (ROOT / "summary/results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "lines.linewidth": 1.1})
    fig, axis = plt.subplots(figsize=(6.25, 3.85), constrained_layout=True)
    colors = {"0": "#1f4e79", "1": "#b55d12", "2": "#2f7d4a"}
    for component in ("0", "1", "2"):
        energy, strength = spectra[component]
        visible = energy >= 4.0
        axis.plot(energy[visible], strength[visible], color=colors[component], label=rf"$K={component}$")
    axis.set_xlim(0.0, 35.0)
    plotted_peak = max(float(np.max(strength[energy >= 5.0])) for energy, strength in spectra.values())
    axis.set_ylim(0.0, 1.15 * plotted_peak)
    axis.set_xlabel("Excitation energy (MeV)")
    axis.set_ylabel(r"Isoscalar E2 strength (fm$^4$ MeV$^{-1}$)")
    axis.grid(color="0.9", linewidth=0.45)
    axis.legend(frameon=False, ncol=3)
    fig.savefig(ROOT / "figures/20ne_kresolved_e2.pdf")
    fig.savefig(ROOT / "figures/20ne_kresolved_e2.png", dpi=240)
    plt.close(fig)

    time, base = normalized_time_signal(ROOT / "td/K0", 5.0e-5, ROOT / "td/reference_K0")
    time_h, half = normalized_time_signal(ROOT / "td/K0_half", 2.5e-5, ROOT / "td/reference_K0")
    fig, axis = plt.subplots(figsize=(6.25, 3.3), constrained_layout=True)
    axis.plot(time, base, color="#1f4e79", label=r"$\eta=5\times10^{-5}$")
    axis.plot(time_h, half, color="#b55d12", linestyle="--", label=r"$\eta=2.5\times10^{-5}$")
    axis.set_xlabel(r"Time (fm/$c$)")
    axis.set_ylabel(r"Reference-subtracted $\Delta Q_{20}/\eta$ (fm$^4$)")
    axis.grid(color="0.9", linewidth=0.45)
    axis.legend(frameon=False)
    fig.savefig(ROOT / "figures/k0_linearity_time_domain.pdf")
    fig.savefig(ROOT / "figures/k0_linearity_time_domain.png", dpi=240)
    plt.close(fig)


if __name__ == "__main__":
    main()
