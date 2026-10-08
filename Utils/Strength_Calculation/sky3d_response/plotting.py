"""Publication-oriented plots generated directly from response products."""

from __future__ import annotations

from collections import defaultdict
from math import ceil
from pathlib import Path
import re
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .models import ResponseConfig, Spectrum
from .output import slugify


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.35,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def _window_label(spectrum: Spectrum) -> str:
    if spectrum.window.kind == "exponential":
        return rf"$\Gamma_{{\rm sm}}={spectrum.window.gamma_mev:g}$ MeV"
    if spectrum.window.kind == "cosine":
        return rf"$\cos^{{{spectrum.window.power}}}$ window"
    return "no window"


def _ylabel(spectrum: Spectrum) -> str:
    units = re.sub(
        r"([A-Za-z]+)\^(-?\d+)",
        lambda match: rf"{match.group(1)}$^{{{match.group(2)}}}$",
        spectrum.channel.strength_units,
    )
    if spectrum.channel.b_elambda:
        return rf"$dB(E{spectrum.channel.multipolarity})/dE$ [{units}]"
    if spectrum.channel.diagonal:
        return rf"$S(E)$ [{units}]"
    return rf"signed $S_{{AF}}(E)$ [{units}]"


def _save(fig: plt.Figure, base: Path, formats: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    metadata = {"Creator": "sky3d-response", "CreationDate": None, "ModDate": None}
    if "pdf" in formats:
        path = base.with_suffix(".pdf")
        fig.savefig(path, metadata=metadata)
        paths.append(path)
    if "png" in formats:
        path = base.with_suffix(".png")
        fig.savefig(path, dpi=300, metadata={"Software": "sky3d-response"})
        paths.append(path)
    return paths


def _plot_channel(
    config: ResponseConfig, spectra: list[Spectrum], output_directory: Path
) -> list[Path]:
    first = spectra[0]
    has_photoabsorption = first.channel.b_elambda and first.channel.multipolarity == 1
    rows = 2 if has_photoabsorption else 1
    fig, axes = plt.subplots(
        rows,
        1,
        figsize=(6.4, 5.4 if rows == 2 else 3.7),
        sharex=rows == 2,
        constrained_layout=True,
        squeeze=False,
    )
    strength_axis = axes[0, 0]
    for spectrum in spectra:
        mask = (spectrum.energy_mev >= config.energy_min_mev) & (
            spectrum.energy_mev <= config.energy_max_mev
        )
        strength_axis.plot(
            spectrum.energy_mev[mask],
            spectrum.signed_strength[mask],
            label=_window_label(spectrum),
        )
    if not first.channel.diagonal:
        strength_axis.axhline(0.0, color="0.35", linewidth=0.7)
    strength_axis.set_ylabel(_ylabel(first))
    strength_axis.set_title(first.channel.label or first.channel.name)
    strength_axis.legend(frameon=False)
    strength_axis.grid(alpha=0.18, linewidth=0.5)

    if has_photoabsorption:
        photo_axis = axes[1, 0]
        for spectrum in spectra:
            mask = (spectrum.energy_mev >= config.energy_min_mev) & (
                spectrum.energy_mev <= config.energy_max_mev
            )
            photo_axis.plot(
                spectrum.energy_mev[mask],
                spectrum.photoabsorption_mb[mask],
                label=_window_label(spectrum),
            )
        photo_axis.set_ylabel(r"$\sigma_\gamma(E)$ [mb]")
        photo_axis.grid(alpha=0.18, linewidth=0.5)
    axes[-1, 0].set_xlabel("Excitation energy [MeV]")
    axes[-1, 0].set_xlim(config.energy_min_mev, config.energy_max_mev)
    fig.text(
        0.995,
        0.005,
        (
            rf"$T={first.propagation_time_fm_c:g}$ fm/$c$, "
            rf"$\Delta E_{{\rm Rayleigh}}={first.rayleigh_resolution_mev:.3g}$ MeV"
        ),
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="0.3",
    )
    return _save(
        fig,
        output_directory / f"spectrum_{slugify(first.channel.name)}",
        config.output_formats,
    )


def _plot_overview(
    config: ResponseConfig,
    grouped: dict[str, list[Spectrum]],
    output_directory: Path,
) -> list[Path]:
    count = len(grouped)
    columns = 2 if count > 1 else 1
    rows = ceil(count / columns)
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(6.8, 2.75 * rows + 0.35),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, spectra in zip(axes.flat, grouped.values()):
        first = spectra[0]
        for spectrum in spectra:
            mask = (spectrum.energy_mev >= config.energy_min_mev) & (
                spectrum.energy_mev <= config.energy_max_mev
            )
            axis.plot(
                spectrum.energy_mev[mask],
                spectrum.signed_strength[mask],
                label=_window_label(spectrum),
            )
        if not first.channel.diagonal:
            axis.axhline(0.0, color="0.35", linewidth=0.7)
        axis.set_title(first.channel.label or first.channel.name)
        axis.set_ylabel(_ylabel(first))
        axis.grid(alpha=0.18, linewidth=0.5)
        axis.legend(frameon=False)
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    for axis in axes[-1, :]:
        if axis.get_visible():
            axis.set_xlabel("Excitation energy [MeV]")
    for axis in axes.flat[:count]:
        axis.set_xlim(config.energy_min_mev, config.energy_max_mev)
    return _save(fig, output_directory / "spectra_overview", config.output_formats)


def write_plots(
    config: ResponseConfig, spectra: Iterable[Spectrum], output_directory: Path
) -> list[Path]:
    if "pdf" not in config.output_formats and "png" not in config.output_formats:
        return []
    _style()
    output_directory.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[Spectrum]] = defaultdict(list)
    for spectrum in spectra:
        grouped[spectrum.channel.name].append(spectrum)
    written: list[Path] = []
    for channel_spectra in grouped.values():
        written.extend(_plot_channel(config, channel_spectra, output_directory))
    written.extend(_plot_overview(config, grouped, output_directory))
    plt.close("all")
    return written
