#!/usr/bin/env python3
"""Extract the iThemba 24Mg IS0 points from the author-supplied vector figure.

This is deliberately a digitization workflow, not publisher-supplied numerical
data.  The arXiv source EPS retains the marker centers and vertical error bars
as vector coordinates, so no hand-clicked or visually estimated points enter.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data" / "experimental" / "source" / "fig5_bahini2022_arxiv.eps"
OUTPUT = PROJECT / "data" / "experimental" / "digitized_is0_bahini2022.csv"
METADATA = PROJECT / "data" / "experimental" / "digitized_is0_bahini2022.json"

EXPECTED_SHA256 = "92009487c140d8b8169de8fe3810f281619df0dfc014f021bb6dab77ae3fdba3"

# Axis calibration read directly from the EPS major ticks in the upper panel.
X_AT_10_MEV = 91.363
X_AT_15_MEV = 196.492
Y_AT_0 = 419.455
Y_AT_20 = 354.907
X_PER_MEV = (X_AT_15_MEV - X_AT_10_MEV) / 5.0
Y_PER_STRENGTH = (Y_AT_0 - Y_AT_20) / 20.0

# The filled-circle path starts at its rightmost point.  This radius is encoded
# identically for all upper-panel iThemba markers.
MARKER_RADIUS = 3.746

CIRCLE = re.compile(
    r"(?P<right>\d+\.\d+) (?P<y>\d+\.\d+) m "
    r"(?P=right) [^\n]+(?:\n[^\n]+)*? h\s+"
    r"(?P=right) (?P=y) m f",
    re.MULTILINE,
)
VERTICAL_SEGMENT = re.compile(
    r"q 1 0 0 0\.99816 0 0 cm\s+"
    r"(?P<x>\d+\.\d+) (?P<y1>\d+\.\d+) m "
    r"(?P=x) (?P<y2>\d+\.\d+) l S Q"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def energy_from_x(x: float) -> float:
    return 10.0 + (x - X_AT_10_MEV) / X_PER_MEV


def strength_from_y(y: float) -> float:
    return (Y_AT_0 - y) / Y_PER_STRENGTH


def main() -> None:
    actual_hash = sha256(SOURCE)
    if actual_hash != EXPECTED_SHA256:
        raise ValueError(f"unexpected EPS SHA-256: {actual_hash}")
    text = SOURCE.read_text(encoding="ascii")

    required_axis_tokens = (
        "91.363 419.455 m",
        "196.492 419.455 m",
        "70.34 354.907 m",
        "70.34 419.455 m",
    )
    if any(token not in text for token in required_axis_tokens):
        raise ValueError("upper-panel calibration ticks do not match the audited figure")

    # The first occurrence of the nucleus label follows the first, upper-panel
    # iThemba marker series.  Limiting the search here avoids the duplicated
    # comparison panel and legend sample.
    upper_markers = text[: text.index("(24)Tj")]
    points: dict[float, tuple[float, float]] = {}
    for match in CIRCLE.finditer(upper_markers):
        x = float(match.group("right")) - MARKER_RADIUS
        y = float(match.group("y"))
        energy_raw = energy_from_x(x)
        bin_center = 9.75 + 0.5 * round((energy_raw - 9.75) / 0.5)
        if 9.75 <= bin_center <= 24.75 and abs(energy_raw - bin_center) < 0.002:
            points[bin_center] = (x, y)
    if len(points) != 31:
        raise ValueError(f"expected 31 half-MeV iThemba bins, found {len(points)}")

    # The upper-panel data and its legend finish at the first iThemba label.
    upper_panel = text[: text.index("(\\) iThemba)Tj")]
    segments = [
        (float(item.group("x")), float(item.group("y1")), float(item.group("y2")))
        for item in VERTICAL_SEGMENT.finditer(upper_panel)
    ]

    records: list[dict[str, object]] = []
    for energy in sorted(points):
        x, y = points[energy]
        endpoints = [y]
        for segment_x, y1, y2 in segments:
            if abs(segment_x - x) < 0.002 and all(90.0 <= value <= 425.0 for value in (y1, y2)):
                endpoints.extend((y1, y2))
        if len(endpoints) != 5:
            raise ValueError(
                f"expected two error-bar segments at E={energy:.2f} MeV, "
                f"found {(len(endpoints) - 1) // 2}"
            )
        central = strength_from_y(y)
        upper = strength_from_y(min(endpoints))
        lower = strength_from_y(max(endpoints))
        records.append(
            {
                "energy_mev": energy,
                "energy_bin_width_mev": 0.5,
                "is0_strength_fm4_per_mev": central,
                "experimental_uncertainty_minus_fm4_per_mev": central - lower,
                "experimental_uncertainty_plus_fm4_per_mev": upper - central,
                "digitization_coordinate_uncertainty_fm4_per_mev": 0.2,
                "source_kind": "digitized_from_author_arxiv_vector_figure",
                "source_figure": "Bahini et al. PRC 105 (2022), Fig. 6",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)

    narrow_bin = min(records, key=lambda row: abs(float(row["energy_mev"]) - 13.75))
    validation_area = float(narrow_bin["is0_strength_fm4_per_mev"]) * 0.5
    metadata = {
        "source": {
            "article_doi": "10.1103/PhysRevC.105.024311",
            "arxiv": "2111.07105",
            "arxiv_eprint_url": "https://export.arxiv.org/e-print/2111.07105",
            "figure_in_article": "Fig. 6",
            "file": str(SOURCE.resolve()),
            "sha256": actual_hash,
        },
        "classification": "digitized experimental data; not publisher-supplied numerical data",
        "method": (
            "Filled-circle centers and vertical error-bar endpoints are parsed directly from "
            "the author-supplied Cairo EPS; axis calibration uses its major-tick coordinates."
        ),
        "binning_mev": 0.5,
        "reported_experimental_resolution_fwhm_kev": 70.0,
        "axis_calibration": {
            "x_at_10_mev": X_AT_10_MEV,
            "x_at_15_mev": X_AT_15_MEV,
            "y_at_zero_strength": Y_AT_0,
            "y_at_20_fm4_per_mev": Y_AT_20,
        },
        "digitization_coordinate_uncertainty_fm4_per_mev": 0.2,
        "validation": {
            "description": (
                "The 13.75-MeV bin area is compared with the independently tabulated "
                "13.87-MeV state strength of 37.7(3.8) fm^4."
            ),
            "digitized_13p75_bin_area_fm4": validation_area,
            "tabulated_13p87_state_strength_fm4": 37.7,
            "tabulated_13p87_state_uncertainty_fm4": 3.8,
        },
    }
    METADATA.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} digitized bins to {OUTPUT}")
    print(f"13.75-MeV bin area: {validation_area:.4f} fm^4")


if __name__ == "__main__":
    main()
