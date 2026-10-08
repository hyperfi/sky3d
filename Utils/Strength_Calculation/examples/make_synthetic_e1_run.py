#!/usr/bin/env python3
"""Create a clearly labeled analytic E1 response for an end-to-end demo."""

import argparse
from pathlib import Path

import numpy as np

HBARC = 197.3269804
BOOST = 5.0e-5
MODES = ((12.0, 5.0), (18.0, 3.0))  # (energy [MeV], B(E1) [e^2 fm^2])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "synthetic_e1_run",
    )
    output = parser.parse_args().output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    time = np.arange(0.0, 4000.0 + 2.0, 2.0)
    signal = np.full_like(time, 10.0)
    for energy, strength in MODES:
        signal += -2.0 * BOOST * strength * np.sin(energy * time / HBARC)

    with (output / "extfield.res").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# time[fm/c] synthetic electric-dipole expectation\n")
        for sample_time, sample_signal in zip(time, signal):
            handle.write(f"{sample_time:14.6f} {sample_signal:24.15e}\n")
    (output / "for005").write_text(
        " &force name='SYNTHETIC-ANALYTIC', pairing='NONE' /\n"
        " &dynamic nt=2000, dt=2.0, texternal=T /\n"
        " &extern ipulse=0, isoext=0, ampl_ext=5D-5, L_val=1, M_val=0, "
        "only_P=1 /\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
