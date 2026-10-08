#!/usr/bin/env python3
"""Create a compact review report for the completed two-EDF E0 gate."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "data" / "processed" / "highres"
TARGET_DURATION = 18000.0
TARGET_GAMMA = 0.2


def rows(name: str) -> list[dict[str, str]]:
    with (OUTPUT / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def one(data: list[dict[str, str]], **conditions: object) -> dict[str, str]:
    selected = []
    for row in data:
        matched = True
        for key, expected in conditions.items():
            actual = row[key]
            if isinstance(expected, float):
                matched = matched and math.isclose(float(actual), expected)
            else:
                matched = matched and actual == expected
        if matched:
            selected.append(row)
    if len(selected) != 1:
        raise ValueError(f"expected one row for {conditions}; found {len(selected)}")
    return selected[0]


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> None:
    summaries = rows("summary.csv")
    stability = rows("stability.csv")
    convergence = rows("time_convergence.csv")
    report: dict[str, object] = {
        "review_status": "requires scientific review before writing e0_stage_go.flag",
        "propagation_time_fm_c": TARGET_DURATION,
        "artificial_smoothing_fwhm_mev": TARGET_GAMMA,
        "rayleigh_resolution_mev": 2.0 * math.pi * 197.3269804 / TARGET_DURATION,
        "edfs": {},
    }

    for slug, edf in (("sly5", "SLy5"), ("skms", "SkM*")):
        family = f"{slug}_n24_e0"
        e0_mqc = one(
            summaries,
            family=family,
            channel="e0_diagonal",
            region="mqc",
            propagation_time_fm_c=TARGET_DURATION,
            artificial_smoothing_fwhm_mev=TARGET_GAMMA,
        )
        e0_main = one(
            summaries,
            family=family,
            channel="e0_diagonal",
            region="main_isgmr",
            propagation_time_fm_c=TARGET_DURATION,
            artificial_smoothing_fwhm_mev=TARGET_GAMMA,
        )
        cross_mqc = one(
            summaries,
            family=family,
            channel="q20_from_e0",
            region="mqc",
            propagation_time_fm_c=TARGET_DURATION,
            artificial_smoothing_fwhm_mev=TARGET_GAMMA,
        )
        reference_stability = one(
            stability,
            family=family,
            role="reference",
            propagation_time_fm_c=TARGET_DURATION,
        )
        boosted_stability = one(
            stability,
            family=family,
            role="e0",
            propagation_time_fm_c=TARGET_DURATION,
        )
        duration_check = one(
            convergence,
            family=family,
            channel="q20_from_e0",
            gamma_mev=TARGET_GAMMA,
            propagation_time_fm_c=12000.0,
        )
        report["edfs"][edf] = {
            "mqc_e0_centroid_mev": number(e0_mqc, "centroid_m1_m0_mev"),
            "mqc_e0_peak_mev": number(e0_mqc, "peak_energy_mev"),
            "mqc_e0_m0_fm4": number(e0_mqc, "m0"),
            "mqc_e0_ewsr_percent": number(e0_mqc, "ewsr_exhaustion_percent"),
            "mqc_cross_characteristic_energy_mev": number(
                cross_mqc, "peak_energy_mev"
            ),
            "mqc_cross_signed_m0_fm4": number(cross_mqc, "m0"),
            "main_isgmr_centroid_mev": number(e0_main, "centroid_m1_m0_mev"),
            "main_isgmr_peak_mev": number(e0_main, "peak_energy_mev"),
            "reference_energy_max_abs_drift_mev": number(
                reference_stability, "energy_max_abs_drift_mev"
            ),
            "reference_q00_max_abs_change_fm2": number(
                reference_stability, "q00_max_abs_change_fm2"
            ),
            "reference_q20_max_abs_change_fm2": number(
                reference_stability, "q20_max_abs_change_fm2"
            ),
            "reference_boundary_shell_particles_max": number(
                reference_stability, "boundary_shell_particles_max"
            ),
            "reference_outermost_max_density_fm3": number(
                reference_stability, "outermost_max_density_max_fm3"
            ),
            "boosted_energy_max_abs_drift_mev": number(
                boosted_stability, "energy_max_abs_drift_mev"
            ),
            "cross_T12000_vs_T18000_relative_l2_9_25": number(
                duration_check, "signed_strength_symmetric_relative_l2_9_25"
            ),
        }

    (OUTPUT / "e0_stage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
