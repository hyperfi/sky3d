"""Plot the CPU and GPU isoscalar quadrupole response from the same restart."""
import argparse
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
source = next(parent for parent in root.parents
              if (parent / "Utils/Strength_Calculation/sky3d_response").is_dir())
sys.path.insert(0, str(source / "Utils/Strength_Calculation"))

from sky3d_response.analysis import analyze_signal
from sky3d_response.models import ChannelSpec, WindowSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=root)
    parser.add_argument("--output", type=Path, default=root / "response")
    parser.add_argument("--end-time", type=float, default=3000.0)
    args = parser.parse_args()

    start = root / "shared-prefix/quadrupoles.res"
    if not start.is_file():
        start = root.parent / "pb208-screen.8bAD7z/grid-40-cpu/quadrupoles.res"
    prefix = np.loadtxt(start, ndmin=2)
    if not np.allclose(prefix[:, 0], [0.0, 2.0, 4.0], rtol=0, atol=1e-8):
        raise ValueError("Unexpected shared prefix samples")

    gam = 1.0
    channel = ChannelSpec("IS_Q20", Path("quadrupoles.res"), 1, 2,
                          strength_units="fm^4 / MeV")
    window = WindowSpec("gamma_1", gamma_mev=gam)
    spectra = {}
    for name in ("cpu", "gpu"):
        run = args.runs / name
        text = (run / "for005").read_text()
        boost = re.search(r"\bampl_ext\s*=\s*([+\-.\dDdEe]+)", text)
        if not boost or not re.search(r"\bL_val\s*=\s*2,\s*M_val\s*=\s*0\b", text):
            raise ValueError(f"Unexpected quadrupole boost in {run}")
        amplitude = float(boost[1].replace("D", "E").replace("d", "e"))
        if amplitude != 5e-5:
            raise ValueError(f"Unexpected boost amplitude in {run}: {amplitude}")
        data = np.loadtxt(run / "quadrupoles.res", ndmin=2)
        if not np.allclose(data[0], prefix[-1], rtol=0, atol=1e-8):
            raise ValueError(f"Restart quadrupole does not match shared prefix: {run}")
        full = np.vstack((prefix[:-1], data))
        expected = np.arange(0.0, args.end_time + 1e-8, 2.0)
        if full.shape[0] != expected.size or not np.allclose(full[:, 0], expected, rtol=0, atol=1e-8):
            raise ValueError(f"Incomplete or misaligned time series: {run}")
        spectra[name] = analyze_signal(full[:, 0], full[:, 1], channel, window,
                                       boost_amplitude=amplitude, zero_padding_factor=8)

    cpu, gpu = spectra["cpu"], spectra["gpu"]
    energy = cpu.energy_mev
    if not np.array_equal(energy, gpu.energy_mev):
        raise ValueError("CPU and GPU energy axes differ")
    cpu_strength, gpu_strength = cpu.signed_strength, gpu.signed_strength
    delta = gpu_strength - cpu_strength
    mask = energy <= 35.0
    if not all(np.isfinite(x).all() for x in (energy, cpu_strength, gpu_strength, delta)):
        raise ValueError("Spectrum contains non-finite values")

    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "pb208-isq20-gamma1.csv",
               np.column_stack((energy, cpu_strength, gpu_strength, delta)), delimiter=",",
               header="energy_mev,cpu_fm4_per_mev,gpu_fm4_per_mev,gpu_minus_cpu_fm4_per_mev",
               comments="")

    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 11, "savefig.facecolor": "white"})
    fig, (top, residual) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                        height_ratios=[3, 1], layout="constrained")
    top.plot(energy[mask], cpu_strength[mask], color="#174879", lw=2, label="CPU")
    top.plot(energy[mask], gpu_strength[mask], color="#c55721", lw=1.55, ls="--", label="GPU")
    top.set_ylabel(r"Strength (fm$^4$/MeV)")
    top.legend(frameon=False)
    top.set_title(r"$^{208}$Pb isoscalar quadrupole response, $M=0$", loc="left")
    residual.plot(energy[mask], delta[mask], color="#424b57", lw=1.25)
    residual.axhline(0.0, color="#7a858e", lw=0.6)
    residual.set(xlim=(0, 35), xlabel="Energy (MeV)", ylabel="GPU − CPU")
    for axis in (top, residual):
        axis.grid(alpha=0.18)
    fig.suptitle(fr"SLy5 · $40^3$ at 1 fm · {args.end_time:g} fm/c · $\Gamma$ = 1 MeV")
    for extension in ("png", "pdf"):
        fig.savefig(args.output / f"pb208-isq20-gamma1.{extension}", dpi=180)
    plt.close(fig)

    stats = dict(channel="isoscalar quadrupole, L=2, M=0", damping_mev=gam,
                 damping_convention="exp(-Gamma t / (2 hbar)); Gamma is Lorentzian FWHM",
                 interval_fm_c=[0.0, args.end_time], sample_spacing_fm_c=2.0,
                 shared_cpu_prefix_through_fm_c=4.0,
                 unboosted_reference_subtraction=False,
                 plot_energy_range_mev=[0.0, 35.0],
                 max_abs_gpu_minus_cpu_fm4_per_mev=float(np.max(np.abs(delta[mask]))),
                 relative_l2_gpu_minus_cpu=float(np.linalg.norm(delta[mask]) /
                                                 np.linalg.norm(cpu_strength[mask])))
    (args.output / "pb208-isq20-gamma1.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Saved CPU and GPU response plots and spectrum to {args.output}")


if __name__ == "__main__":
    main()

