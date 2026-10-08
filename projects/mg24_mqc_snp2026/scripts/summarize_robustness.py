#!/usr/bin/env python3
"""Assemble smoothing, EWSR, reciprocity, and linearity robustness tables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT / "data" / "processed"
WINDOWS = ("gamma_1", "gamma_2", "cosine_6")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def find(rows: list[dict[str, str]], channel: str, window: str, region: str) -> dict[str, str]:
    for row in rows:
        if row["channel"] == channel and row["window"] == window and row["region"] == region:
            return row
    raise KeyError((channel, window, region))


def value(row: dict[str, str], key: str) -> float:
    return float(row[key])


def json_safe(item):
    if isinstance(item, float) and not math.isfinite(item):
        return None
    if isinstance(item, dict):
        return {key: json_safe(value) for key, value in item.items()}
    if isinstance(item, list):
        return [json_safe(value) for value in item]
    return item


def main() -> None:
    e0 = read_rows(PROCESSED / "e0_baseline" / "summary.csv")
    e2 = read_rows(PROCESSED / "e2k0_baseline" / "summary.csv")
    milestone2 = json.loads((PROCESSED / "milestone2_summary.json").read_text(encoding="utf-8"))
    reciprocity = json.loads((PROCESSED / "reciprocity_summary.json").read_text(encoding="utf-8"))
    linearity = json.loads((PROCESSED / "linearity_summary.json").read_text(encoding="utf-8"))

    records: list[dict[str, object]] = []
    for window in WINDOWS:
        e0_mqc = find(e0, "e0_diagonal", window, "mqc")
        e0_high = find(e0, "e0_diagonal", window, "main_isgmr")
        e0_full = find(e0, "e0_diagonal", window, "experiment_range")
        e2_mqc = find(e2, "e2k0_diagonal", window, "mqc")
        cross_forward = find(e0, "q20_from_e0", window, "mqc")
        cross_reverse = find(e2, "q00_from_e2k0", window, "mqc")
        artificial = value(e0_mqc, "artificial_smoothing_fwhm_mev")
        record = {
            "window": window,
            "edf": milestone2["static_hf"]["edf"],
            "beta2": milestone2["static_hf"]["beta2"],
            "boost_eta": 5.0e-5,
            "time_step_fm_over_c": 0.2,
            "output_sample_spacing_fm_over_c": 2.0,
            "propagation_time_fm_over_c": 2000.0,
            "artificial_smoothing_fwhm_mev": artificial,
            "rayleigh_resolution_mev": 0.6199209919796971,
            "e0_mqc_peak_mev": value(e0_mqc, "peak_energy_mev"),
            "e0_mqc_centroid_mev": value(e0_mqc, "centroid_m1_m0_mev"),
            "e0_mqc_m0_fm4": value(e0_mqc, "m0"),
            "e0_mqc_m1_mev_fm4": value(e0_mqc, "m1"),
            "e0_mqc_ewsr_percent": value(e0_mqc, "ewsr_exhaustion_percent"),
            "e0_observed_fwhm_mev": value(e0_mqc, "observed_fwhm_mev"),
            "e0_lorentzian_intrinsic_estimate_mev": value(
                e0_mqc, "intrinsic_fwhm_lorentzian_estimate_mev"
            ),
            "e0_main_peak_mev": value(e0_high, "peak_energy_mev"),
            "e0_main_centroid_mev": value(e0_high, "centroid_m1_m0_mev"),
            "e0_main_ewsr_percent": value(e0_high, "ewsr_exhaustion_percent"),
            "e0_9_25_centroid_mev": value(e0_full, "centroid_m1_m0_mev"),
            "e0_9_25_ewsr_percent": value(e0_full, "ewsr_exhaustion_percent"),
            "e2k0_mqc_peak_mev": value(e2_mqc, "peak_energy_mev"),
            "e2k0_mqc_centroid_mev": value(e2_mqc, "centroid_m1_m0_mev"),
            "e2k0_mqc_ewsr_percent": value(e2_mqc, "ewsr_exhaustion_percent"),
            "forward_cross_peak_mev": value(cross_forward, "peak_energy_mev"),
            "forward_cross_signed_m0_fm4": value(cross_forward, "m0"),
            "reverse_cross_peak_mev": value(cross_reverse, "peak_energy_mev"),
            "reverse_cross_signed_m0_fm4": value(cross_reverse, "m0"),
            "reciprocity_complex_relative_l2": reciprocity["windows"][window]["mqc"]["complex_response"]["symmetric_relative_l2"],
            "reciprocity_signed_correlation": reciprocity["windows"][window]["mqc"]["signed_strength"]["pearson_correlation"],
            "linearity_e0_complex_relative_l2": linearity["windows"][window]["e0_diagonal"]["mqc"]["complex_response"]["symmetric_relative_l2"],
            "linearity_cross_complex_relative_l2": linearity["windows"][window]["q20_from_e0"]["mqc"]["complex_response"]["symmetric_relative_l2"],
        }
        records.append(record)

    with (PROCESSED / "robustness_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    finite_e0_peaks = [float(row["e0_mqc_peak_mev"]) for row in records]
    finite_cross_peaks = [float(row["forward_cross_peak_mev"]) for row in records]
    summary = {
        "records": records,
        "ranges_across_windows": {
            "e0_mqc_peak_mev": [min(finite_e0_peaks), max(finite_e0_peaks)],
            "forward_cross_peak_mev": [min(finite_cross_peaks), max(finite_cross_peaks)],
            "e0_mqc_centroid_mev": [
                min(float(row["e0_mqc_centroid_mev"]) for row in records),
                max(float(row["e0_mqc_centroid_mev"]) for row in records),
            ],
            "e0_main_peak_mev": [
                min(float(row["e0_main_peak_mev"]) for row in records),
                max(float(row["e0_main_peak_mev"]) for row in records),
            ],
        },
        "width_caveat": (
            "Observed FWHM includes finite-time/window effects. The intrinsic field is only "
            "observed minus Gamma_sm under a Lorentzian-convolution assumption; cosine_6 has no assigned artificial FWHM."
        ),
    }
    (PROCESSED / "robustness_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["ranges_across_windows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
