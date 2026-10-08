"""End-to-end configured response-analysis workflow."""

from __future__ import annotations

import hashlib
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from .analysis import analyze_signal, calculate_summary
from .config import load_config
from .constants import (
    E1_PHOTOABSORPTION_MB,
    E2_MEV_FM,
    HBARC_MEV_FM,
)
from .io import parse_sky3d_input, read_response_file, sha256_file
from .models import ResponseConfig, Spectrum, SummaryRow
from .output import write_csv_outputs, write_hdf5, write_metadata_json
from .plotting import write_plots


@dataclass(frozen=True)
class AnalysisRun:
    config: ResponseConfig
    spectra: tuple[Spectrum, ...]
    summaries: tuple[SummaryRow, ...]
    artifacts: tuple[Path, ...]
    metadata: dict[str, Any]


def _git_commit(directory: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unavailable"


def _package_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
    }
    try:
        import h5py

        versions["h5py"] = h5py.__version__
    except ImportError:
        versions["h5py"] = "not installed"
    return versions


def _metadata(
    config: ResponseConfig,
    input_groups: dict[str, dict[str, Any]],
    source_files: dict[str, dict[str, str]],
) -> dict[str, Any]:
    config_hash = hashlib.sha256(config.config_text.encode("utf-8")).hexdigest()
    return {
        "schema": "sky3d-response-v1",
        "sky3d_checkout": "v1.2 response implementation",
        "git_commit": _git_commit(config.run_directory),
        "config_file": str(config.config_path),
        "config_sha256": config_hash,
        "input_file": str(config.input_file),
        "input_sha256": sha256_file(config.input_file),
        "run_directory": str(config.run_directory),
        "source_files": source_files,
        "sky3d_input": input_groups,
        "software": _package_versions(),
        "constants": {
            "hbar_c_mev_fm": HBARC_MEV_FM,
            "e_squared_mev_fm": E2_MEV_FM,
            "e1_photoabsorption_coefficient_mb": E1_PHOTOABSORPTION_MB,
        },
        "conventions": {
            "sky3d_boost": "psi(0+) = exp(-i eta F) psi(0)",
            "transform": "integral_0^T dt exp(-i E t / hbar) delta<A>(t)",
            "response": "-s * transform/(eta*hbar), where boost=exp(i*s*eta*F)",
            "signed_strength": "Im(response)/pi; never absolute-valued for cross channels",
            "exponential_window": "exp[-Gamma_sm t/(2 hbar)]; Gamma_sm is Lorentzian FWHM",
            "rayleigh_resolution": "2*pi*hbar/T; zero padding does not change it",
            "photoabsorption_e1": "sigma_gamma[mb] = coefficient * E * dB(E1)/dE",
            "dipole_polarizability": "alpha_D[fm^3] = (8*pi/9)*e^2*m_-1",
            "intrinsic_width": (
                "observed FWHM - Gamma_sm, reported only as a Lorentzian-convolution estimate"
            ),
        },
    }


def run_analysis(config: ResponseConfig | Path) -> AnalysisRun:
    if isinstance(config, Path):
        config = load_config(config)
    if not config.input_file.is_file():
        raise FileNotFoundError(f"Sky3D input file not found: {config.input_file}")
    input_groups = parse_sky3d_input(config.input_file)
    extern = input_groups.get("extern", {})
    input_amplitude = extern.get("ampl_ext")
    source_files: dict[str, dict[str, str]] = {}
    spectra: list[Spectrum] = []

    for channel in config.channels:
        if not channel.file.is_file():
            raise FileNotFoundError(f"Response file not found: {channel.file}")
        amplitude = channel.boost_amplitude
        if amplitude is None:
            if input_amplitude is None:
                raise ValueError(
                    f"Channel {channel.name!r} has no boost_amplitude and "
                    f"{config.input_file} contains no &extern ampl_ext"
                )
            amplitude = float(input_amplitude)
        time, signal, header = read_response_file(channel.file, channel.column)
        source_files[channel.name] = {
            "path": str(channel.file),
            "sha256": sha256_file(channel.file),
            "header": header,
        }
        reference_signal = None
        reference_metadata: dict[str, object] = {}
        if channel.reference_file is not None:
            if not channel.reference_file.is_file():
                raise FileNotFoundError(
                    f"Reference response file not found: {channel.reference_file}"
                )
            reference_column = (
                channel.reference_column
                if channel.reference_column is not None
                else channel.column
            )
            reference_time, reference_signal, reference_header = read_response_file(
                channel.reference_file, reference_column
            )
            if not np.array_equal(time, reference_time):
                raise ValueError(
                    f"Reference time grid differs for channel {channel.name!r}"
                )
            reference_hash = sha256_file(channel.reference_file)
            source_files[channel.name].update(
                {
                    "reference_path": str(channel.reference_file),
                    "reference_sha256": reference_hash,
                    "reference_header": reference_header,
                    "reference_column": str(reference_column),
                }
            )
            reference_metadata = {
                "reference_file": str(channel.reference_file),
                "reference_sha256": reference_hash,
                "reference_header": reference_header,
                "reference_column": reference_column,
                "reference_scale": channel.reference_scale,
            }
        for window in config.windows:
            spectra.append(
                analyze_signal(
                    time,
                    signal,
                    channel,
                    window,
                    boost_amplitude=amplitude,
                    zero_padding_factor=config.zero_padding_factor,
                    baseline_points=config.baseline_points,
                    reference_signal=reference_signal,
                    metadata={
                        "source_file": str(channel.file),
                        "source_sha256": source_files[channel.name]["sha256"],
                        "source_header": header,
                        "input_boost_L": extern.get("l_val", "unavailable"),
                        "input_boost_M": extern.get("m_val", "unavailable"),
                        "input_isoext": extern.get("isoext", "unavailable"),
                        **reference_metadata,
                    },
                )
            )

    summaries = [
        calculate_summary(
            spectrum,
            region,
            nucleus_a=config.nucleus_a,
            nucleus_z=config.nucleus_z,
            trk_enhancement=config.trk_enhancement,
        )
        for spectrum in spectra
        for region in config.regions
    ]
    metadata = _metadata(config, input_groups, source_files)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    if "csv" in config.output_formats:
        artifacts.extend(
            write_csv_outputs(
                config.output_directory,
                spectra,
                summaries,
                config.energy_min_mev,
                config.energy_max_mev,
            )
        )
    if "hdf5" in config.output_formats:
        artifacts.append(
            write_hdf5(
                config.output_directory / "response.h5",
                config,
                spectra,
                summaries,
                metadata,
            )
        )
    artifacts.extend(write_plots(config, spectra, config.output_directory))
    artifacts.append(write_metadata_json(config.output_directory, metadata))
    return AnalysisRun(
        config=config,
        spectra=tuple(spectra),
        summaries=tuple(summaries),
        artifacts=tuple(artifacts),
        metadata=metadata,
    )
