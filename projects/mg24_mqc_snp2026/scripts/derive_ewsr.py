#!/usr/bin/env python3
"""Derive IS E0/E2 EWSRs in the exact Sky3D operator convention."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = PROJECT / "runs" / "no_boost_long"
OUTPUT = PROJECT / "data" / "processed"

# SV-bas values from Code/forces.data: neutron, proton hbar^2/(2m_q).
H2M_N = 20.72126
H2M_P = 20.74982


def first_row(path: Path) -> np.ndarray:
    return np.atleast_2d(np.loadtxt(path, comments="#"))[0]


def main() -> None:
    mono = first_row(REFERENCE_RUN / "monopoles.res")
    quad = first_row(REFERENCE_RUN / "quadrupoles.res")
    if mono[0] != 0.0 or quad[0] != 0.0:
        raise ValueError("EWSR derivation requires time-zero moments")

    sqrt_4pi = math.sqrt(4.0 * math.pi)
    mono_n = 0.5 * (mono[1] - mono[2])
    mono_p = 0.5 * (mono[1] + mono[2])
    r2_n = mono_n * sqrt_4pi
    r2_p = mono_p * sqrt_4pi

    q20_n = 0.5 * (quad[1] - quad[2])
    q20_p = 0.5 * (quad[1] + quad[2])
    c20 = 5.0 / (4.0 * math.sqrt(math.pi))

    # F00 = r^2/sqrt(4 pi): |grad F00|^2 = r^2/pi.
    e0_ewsr = (H2M_N * r2_n + H2M_P * r2_p) / math.pi

    # F20 = sqrt(5) r^2 Y20 = C20(2z^2-x^2-y^2).
    # |grad F20|^2 = 25/(4pi)(x^2+y^2+4z^2).
    # With R2=x^2+y^2+z^2 and Q20=C20(3z^2-R2), the last factor is
    # 2 R2 + Q20/C20, which is available from the recorded IS/IV moments.
    grad2_n = 25.0 / (4.0 * math.pi) * (2.0 * r2_n + q20_n / c20)
    grad2_p = 25.0 / (4.0 * math.pi) * (2.0 * r2_p + q20_p / c20)
    e2_ewsr = H2M_N * grad2_n + H2M_P * grad2_p

    result = {
        "source": str(REFERENCE_RUN.resolve()),
        "time_fm_over_c": 0.0,
        "sv_bas_hbar2_over_2m_mev_fm2": {"neutron": H2M_N, "proton": H2M_P},
        "species_integrated_r2_fm2": {"neutron": r2_n, "proton": r2_p},
        "species_q20_code_fm2": {"neutron": q20_n, "proton": q20_p},
        "e0": {
            "operator": "F00 = sum_i r_i^2/sqrt(4 pi)",
            "formula": "m1 = sum_q hbar^2/(2m_q) integral rho_q |grad f00|^2",
            "reference_m1_mev_fm4": e0_ewsr,
        },
        "e2k0": {
            "operator": "F20 = sum_i sqrt(5) r_i^2 Y20 = 5/(4sqrt(pi))(2z^2-x^2-y^2)",
            "formula": "m1 = sum_q hbar^2/(2m_q) integral rho_q |grad f20|^2",
            "reference_m1_mev_fm4": e2_ewsr,
        },
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "ewsr_references.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (OUTPUT / "ewsr_references.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["channel", "m1_reference", "unit", "operator"])
        writer.writerow(["e0", f"{e0_ewsr:.12g}", "MeV fm^4", result["e0"]["operator"]])
        writer.writerow(["e2k0", f"{e2_ewsr:.12g}", "MeV fm^4", result["e2k0"]["operator"]])
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

