"""Typed configuration and result containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class WindowSpec:
    name: str
    kind: str = "exponential"
    gamma_mev: float | None = 1.0
    power: int | None = None


@dataclass(frozen=True)
class RegionSpec:
    name: str
    min_mev: float
    max_mev: float


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    file: Path
    column: int
    multipolarity: int
    projection: int = 0
    channel_type: str = "isoscalar"
    diagonal: bool = True
    label: str | None = None
    strength_units: str = "unspecified"
    boost_amplitude: float | None = None
    boost_phase_sign: int = -1
    observable_scale: float = 1.0
    boost_operator_scale: float = 1.0
    b_elambda: bool = False
    ewsr_reference: float | None = None
    ewsr_label: str | None = None
    use_trk_ewsr: bool = False
    reference_file: Path | None = None
    reference_column: int | None = None
    reference_scale: float = 1.0


@dataclass(frozen=True)
class ResponseConfig:
    config_path: Path
    config_text: str
    run_directory: Path
    input_file: Path
    output_directory: Path
    channels: tuple[ChannelSpec, ...]
    windows: tuple[WindowSpec, ...]
    regions: tuple[RegionSpec, ...]
    energy_min_mev: float = 0.0
    energy_max_mev: float = 40.0
    zero_padding_factor: int = 8
    baseline_points: int = 1
    nucleus_a: int | None = None
    nucleus_z: int | None = None
    trk_enhancement: float = 0.0
    output_formats: tuple[str, ...] = ("csv", "hdf5", "pdf", "png")


@dataclass
class Spectrum:
    channel: ChannelSpec
    window: WindowSpec
    time_fm_c: np.ndarray
    raw_signal: np.ndarray
    reference_signal: np.ndarray
    reference_delta: np.ndarray
    delta_signal: np.ndarray
    window_values: np.ndarray
    filtered_signal: np.ndarray
    energy_mev: np.ndarray
    response: np.ndarray
    signed_strength: np.ndarray
    b_elambda_distribution: np.ndarray
    photoabsorption_mb: np.ndarray
    sample_dt_fm_c: float
    propagation_time_fm_c: float
    rayleigh_resolution_mev: float
    energy_bin_mev: float
    boost_amplitude: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SummaryRow:
    channel: str
    window: str
    region: str
    energy_min_mev: float
    energy_max_mev: float
    m_minus1: float
    m0: float
    m1: float
    centroid_m1_m0_mev: float
    peak_energy_mev: float
    observed_fwhm_mev: float
    artificial_smoothing_fwhm_mev: float
    intrinsic_fwhm_lorentzian_estimate_mev: float
    alpha_d_fm3: float
    ewsr_reference: float
    ewsr_exhaustion_percent: float
    negative_area_fraction: float
    diagonal: bool
    strength_units: str
    ewsr_label: str
