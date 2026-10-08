#!/usr/bin/env python3
"""Fill the concise SLy5/SkM* comparison table from high-resolution summaries."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = PROJECT / "data" / "processed" / "two_edf_quantitative_summary.csv"
DEFAULT_SUMMARY = PROJECT / "data" / "processed" / "highres" / "summary.csv"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    result.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    result.add_argument("--duration", type=float, default=18000.0)
    result.add_argument("--gamma", type=float, default=0.2)
    return result


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def select(
    rows: list[dict[str, str]],
    *,
    family: str,
    channel: str,
    region: str,
    duration: float,
    gamma: float,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row["family"] == family
        and row["channel"] == channel
        and row["region"] == region
        and math.isclose(float(row["propagation_time_fm_c"]), duration)
        and math.isclose(float(row["artificial_smoothing_fwhm_mev"]), gamma)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one summary row for {family}/{channel}/{region} at "
            f"T={duration:g}, Gamma={gamma:g}; found {len(matches)}"
        )
    return matches[0]


def main() -> None:
    arguments = parser().parse_args()
    summaries = read_rows(arguments.summary)
    static_rows = read_rows(arguments.table)
    static_by_edf = {row["edf"]: row for row in static_rows}
    output_rows: list[dict[str, str]] = []

    for slug, edf in (("sly5", "SLy5"), ("skms", "SkM*")):
        e0_mqc = select(
            summaries,
            family=f"{slug}_n24_e0",
            channel="e0_diagonal",
            region="mqc",
            duration=arguments.duration,
            gamma=arguments.gamma,
        )
        e0_main = select(
            summaries,
            family=f"{slug}_n24_e0",
            channel="e0_diagonal",
            region="main_isgmr",
            duration=arguments.duration,
            gamma=arguments.gamma,
        )
        e2_mqc = select(
            summaries,
            family=f"{slug}_n24_e2",
            channel="e2k0_diagonal",
            region="mqc",
            duration=arguments.duration,
            gamma=arguments.gamma,
        )
        cross_mqc = select(
            summaries,
            family=f"{slug}_n24_e0",
            channel="q20_from_e0",
            region="mqc",
            duration=arguments.duration,
            gamma=arguments.gamma,
        )
        row = dict(static_by_edf[edf])
        row.update(
            {
                "mqc_e0_centroid_mev": e0_mqc["centroid_m1_m0_mev"],
                "mqc_e2k0_centroid_mev": e2_mqc["centroid_m1_m0_mev"],
                "cross_characteristic_energy_mev": cross_mqc["peak_energy_mev"],
                "main_isgmr_centroid_mev": e0_main["centroid_m1_m0_mev"],
                "mqc_e0_m0_fm4": e0_mqc["m0"],
                "mqc_e0_ewsr_percent": e0_mqc["ewsr_exhaustion_percent"],
                "response_status": (
                    f"complete at T={arguments.duration:g} fm/c, "
                    f"Gamma={arguments.gamma:g} MeV"
                ),
            }
        )
        output_rows.append(row)

    fieldnames = list(static_rows[0])
    with arguments.table.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)


if __name__ == "__main__":
    main()
