"""Reproducible linear-response analysis for Sky3D time signals."""

from .analysis import analyze_signal, calculate_summary, make_window
from .config import load_config
from .density import DensitySnapshot, density_diagnostics, read_tdd_density
from .io import parse_sky3d_input, read_response_file
from .models import ChannelSpec, RegionSpec, ResponseConfig, Spectrum, WindowSpec

__all__ = [
    "ChannelSpec",
    "DensitySnapshot",
    "RegionSpec",
    "ResponseConfig",
    "Spectrum",
    "WindowSpec",
    "analyze_signal",
    "calculate_summary",
    "density_diagnostics",
    "load_config",
    "make_window",
    "parse_sky3d_input",
    "read_response_file",
    "read_tdd_density",
]

__version__ = "1.0.0"
