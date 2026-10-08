#!/usr/bin/env python3
"""Create reproducible static-HF and stationarity summaries and plots."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
STATIC_RUN = PROJECT / "runs" / "static_svbas"
STATIC_LOG = PROJECT / "logs" / "static_svbas.log"
ZSEED_RUN = PROJECT / "runs" / "static_svbas_zseed"
STATIONARY_RUN = PROJECT / "runs" / "stationarity_zaxis"
STATIONARY_LOG = PROJECT / "logs" / "stationarity_zaxis.log"
PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"


FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"


def load_table(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, comments="#"))


def last_match(text: str, pattern: str) -> float:
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    if not matches:
        raise ValueError(f"pattern not found: {pattern}")
    return float(matches[-1].replace("D", "E").replace("d", "e"))


def moment_rows(text: str) -> np.ndarray:
    rows: list[list[float]] = []
    for line in text.splitlines():
        if re.match(r"^\s+Total:\s+", line):
            values = [float(value.replace("D", "E")) for value in re.findall(FLOAT, line)]
            if len(values) == 9:
                rows.append(values)
    if not rows:
        raise ValueError("no total moment rows found")
    return np.asarray(rows)


def drift(values: np.ndarray) -> dict[str, float]:
    initial = float(values[0])
    final = float(values[-1])
    absolute = np.abs(values - initial)
    scale = abs(initial)
    return {
        "initial": initial,
        "final": final,
        "final_minus_initial": final - initial,
        "max_abs_from_initial": float(np.max(absolute)),
        "peak_to_peak": float(np.ptp(values)),
        "max_relative_from_initial": float(np.max(absolute) / scale) if scale else float("nan"),
    }


def write_metric_csv(path: Path, rows: list[tuple[str, float | int | str, str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["metric", "value", "unit", "source"])
        writer.writerows(rows)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    conver = load_table(STATIC_RUN / "conver.res")
    zseed = load_table(ZSEED_RUN / "conver.res")
    static_text = STATIC_LOG.read_text(encoding="utf-8", errors="replace")
    static_moment = moment_rows(static_text)[-1]

    static = {
        "edf": "SV-bas",
        "pairing": "NONE",
        "grid": {"nx": 24, "ny": 24, "nz": 24, "dx_fm": 1.0, "dy_fm": 1.0, "dz_fm": 1.0},
        "boundary": "nonperiodic",
        "energy_mev": last_match(static_text, rf"Total energy:\s*({FLOAT})\s*MeV"),
        "rms_total_fm": float(static_moment[1]),
        "rms_neutron_fm": last_match(static_text, rf"^\s*Neutron:\s+{FLOAT}\s+({FLOAT})"),
        "rms_proton_fm": last_match(static_text, rf"^\s*Proton:\s+{FLOAT}\s+({FLOAT})"),
        "intrinsic_q20_fm2": float(static_moment[2]),
        "mean_square_axes_fm2": {
            "x": float(static_moment[3]),
            "y": float(static_moment[4]),
            "z": float(static_moment[5]),
        },
        "beta2": float(conver[-1, 9]),
        "gamma_deg": float(conver[-1, 10]),
        "raw_symmetry_axis": "x",
        "production_symmetry_axis": "z after exact R_y(-pi/2) grid/spinor rotation",
        "iterations": int(conver[-1, 0]),
        "requested_fluctuation_tolerance_mev": 1.0e-6,
        "h2_fluctuation_floor_mev": float(conver[-1, 3]),
        "hh_fluctuation_floor_mev": float(conver[-1, 4]),
        "max_iteration_stop": True,
        "seed_check": {
            "second_seed_iterations": int(zseed[-1, 0]),
            "energy_difference_at_printed_precision_mev": float(zseed[-1, 1] - conver[-1, 1]),
            "beta2_difference": float(zseed[-1, 9] - conver[-1, 9]),
            "rms_difference_fm": float(zseed[-1, 5] - conver[-1, 5]),
        },
    }

    energies = load_table(STATIONARY_RUN / "energies.res")
    monopoles = load_table(STATIONARY_RUN / "monopoles.res")
    quadrupoles = load_table(STATIONARY_RUN / "quadrupoles.res")
    stationary_text = STATIONARY_LOG.read_text(encoding="utf-8", errors="replace")
    moments = moment_rows(stationary_text)
    cm_norm = np.linalg.norm(moments[:, 6:9], axis=1)

    stationarity = {
        "duration_fm_over_c": float(energies[-1, 0]),
        "dt_fm_over_c": 0.2,
        "samples": int(energies.shape[0]),
        "energy_sum_mev": drift(energies[:, 3]),
        "energy_integrated_mev": drift(energies[:, 4]),
        "neutron_number": drift(energies[:, 1]),
        "proton_number": drift(energies[:, 2]),
        "monopole_is_fm2": drift(monopoles[:, 1]),
        "quadrupole_k0_is_code_units": drift(quadrupoles[:, 1]),
        "center_of_mass_norm_max_fm": float(np.max(cm_norm)),
    }

    summary = {"static_hf": static, "stationarity": stationarity}
    (PROCESSED / "milestone2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    write_metric_csv(
        PROCESSED / "static_summary.csv",
        [
            ("energy", static["energy_mev"], "MeV", "static_svbas.log"),
            ("rms_total", static["rms_total_fm"], "fm", "static_svbas.log"),
            ("rms_neutron", static["rms_neutron_fm"], "fm", "static_svbas.log"),
            ("rms_proton", static["rms_proton_fm"], "fm", "static_svbas.log"),
            ("intrinsic_q20", static["intrinsic_q20_fm2"], "fm^2", "static_svbas.log"),
            ("beta2", static["beta2"], "1", "conver.res"),
            ("gamma", static["gamma_deg"], "degree", "conver.res"),
            ("h2_fluctuation_floor", static["h2_fluctuation_floor_mev"], "MeV", "conver.res"),
            ("hh_fluctuation_floor", static["hh_fluctuation_floor_mev"], "MeV", "conver.res"),
        ],
    )
    write_metric_csv(
        PROCESSED / "stationarity_summary.csv",
        [
            ("duration", stationarity["duration_fm_over_c"], "fm/c", "energies.res"),
            ("energy_max_abs_drift", stationarity["energy_sum_mev"]["max_abs_from_initial"], "MeV", "energies.res"),
            ("neutron_number_max_abs_drift", stationarity["neutron_number"]["max_abs_from_initial"], "1", "energies.res"),
            ("proton_number_max_abs_drift", stationarity["proton_number"]["max_abs_from_initial"], "1", "energies.res"),
            ("monopole_max_relative_drift", stationarity["monopole_is_fm2"]["max_relative_from_initial"], "1", "monopoles.res"),
            ("quadrupole_max_relative_drift", stationarity["quadrupole_k0_is_code_units"]["max_relative_from_initial"], "1", "quadrupoles.res"),
            ("center_of_mass_norm_max", stationarity["center_of_mass_norm_max_fm"], "fm", "stationarity_zaxis.log"),
        ],
    )

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 220})
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 5.4), sharex=True, constrained_layout=True)
    axes[0].plot(energies[:, 0], (energies[:, 3] - energies[0, 3]) * 1.0e6, color="#1f4e79", lw=1.4)
    axes[0].axhline(0.0, color="0.45", lw=0.6)
    axes[0].set_ylabel(r"$E(t)-E(0)$ (eV)")
    axes[0].set_title(r"Unboosted $^{24}$Mg TDHF stationarity (SV-bas)")

    axes[1].plot(
        monopoles[:, 0],
        (monopoles[:, 1] / monopoles[0, 1] - 1.0) * 1.0e6,
        label=r"$Q_{00}$",
        color="#b44b27",
        lw=1.35,
    )
    axes[1].plot(
        quadrupoles[:, 0],
        (quadrupoles[:, 1] / quadrupoles[0, 1] - 1.0) * 1.0e6,
        label=r"$Q_{20}$ (intrinsic axis)",
        color="#2f7d4a",
        lw=1.35,
    )
    axes[1].axhline(0.0, color="0.45", lw=0.6)
    axes[1].set_xlabel(r"Time (fm/$c$)")
    axes[1].set_ylabel("Relative drift (ppm)")
    axes[1].legend(frameon=False, ncol=2)
    for axis in axes:
        axis.tick_params(direction="in", top=True, right=True)

    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"stationarity_svbas.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

