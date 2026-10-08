"""TOML configuration loading and validation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .models import ChannelSpec, RegionSpec, ResponseConfig, WindowSpec


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _require(table: dict[str, Any], key: str, context: str) -> Any:
    if key not in table:
        raise ValueError(f"Missing required key {context}.{key}")
    return table[key]


def load_config(path: Path) -> ResponseConfig:
    path = path.resolve()
    text = path.read_text(encoding="utf-8")
    raw = tomllib.loads(text)
    base = path.parent

    run = raw.get("run", {})
    output = raw.get("output", {})
    analysis = raw.get("analysis", {})
    nucleus = raw.get("nucleus", {})
    run_directory = _resolve(base, str(_require(run, "directory", "run")))
    input_file = _resolve(run_directory, str(run.get("input", "for005")))
    output_directory = _resolve(base, str(output.get("directory", "response_output")))

    default_windows = [
        {"name": "gamma1", "kind": "exponential", "gamma_mev": 1.0}
    ]
    windows = tuple(
        WindowSpec(
            name=str(_require(item, "name", "windows[]")),
            kind=str(item.get("kind", "exponential")).lower(),
            gamma_mev=float(item["gamma_mev"]) if "gamma_mev" in item else None,
            power=int(item["power"]) if "power" in item else None,
        )
        for item in raw.get("windows", default_windows)
    )
    default_regions = [
        {
            "name": "analysis_range",
            "min_mev": analysis.get("energy_min_mev", 0.0),
            "max_mev": analysis.get("energy_max_mev", 40.0),
        }
    ]
    regions = tuple(
        RegionSpec(
            name=str(_require(item, "name", "regions[]")),
            min_mev=float(_require(item, "min_mev", "regions[]")),
            max_mev=float(_require(item, "max_mev", "regions[]")),
        )
        for item in raw.get("regions", default_regions)
    )

    channels_list: list[ChannelSpec] = []
    for item in raw.get("channels", []):
        channel_type = str(item.get("type", "isoscalar")).lower()
        b_elambda = bool(item.get("b_elambda", False))
        diagonal = bool(item.get("diagonal", True))
        if b_elambda and (channel_type != "electric" or not diagonal):
            raise ValueError(
                "b_elambda=true requires type='electric' and diagonal=true; "
                "cross or nuclear-mass responses must not be labeled B(E lambda)"
            )
        boost_phase_sign = int(item.get("boost_phase_sign", -1))
        if boost_phase_sign not in {-1, 1}:
            raise ValueError("boost_phase_sign must be -1 or +1")
        channels_list.append(
            ChannelSpec(
                name=str(_require(item, "name", "channels[]")),
                file=_resolve(run_directory, str(_require(item, "file", "channels[]"))),
                column=int(item.get("column", 1)),
                multipolarity=int(_require(item, "multipolarity", "channels[]")),
                projection=int(item.get("projection", 0)),
                channel_type=channel_type,
                diagonal=diagonal,
                label=str(item["label"]) if "label" in item else None,
                strength_units=str(item.get("strength_units", "unspecified")),
                boost_amplitude=(
                    float(item["boost_amplitude"]) if "boost_amplitude" in item else None
                ),
                boost_phase_sign=boost_phase_sign,
                observable_scale=float(item.get("observable_scale", 1.0)),
                boost_operator_scale=float(item.get("boost_operator_scale", 1.0)),
                b_elambda=b_elambda,
                ewsr_reference=(
                    float(item["ewsr_reference"]) if "ewsr_reference" in item else None
                ),
                ewsr_label=str(item["ewsr_label"]) if "ewsr_label" in item else None,
                use_trk_ewsr=bool(item.get("use_trk_ewsr", False)),
                reference_file=(
                    _resolve(run_directory, str(item["reference_file"]))
                    if "reference_file" in item
                    else None
                ),
                reference_column=(
                    int(item["reference_column"])
                    if "reference_column" in item
                    else None
                ),
                reference_scale=float(item.get("reference_scale", 1.0)),
            )
        )
    if not channels_list:
        raise ValueError("At least one [[channels]] entry is required")
    if len({item.name for item in channels_list}) != len(channels_list):
        raise ValueError("Channel names must be unique")
    if len({item.name for item in windows}) != len(windows):
        raise ValueError("Window names must be unique")
    if len({item.name for item in regions}) != len(regions):
        raise ValueError("Region names must be unique")

    formats = tuple(
        str(value).lower()
        for value in output.get("formats", ["csv", "hdf5", "pdf", "png"])
    )
    allowed_formats = {"csv", "hdf5", "pdf", "png"}
    unknown = set(formats) - allowed_formats
    if unknown:
        raise ValueError(f"Unsupported output formats: {sorted(unknown)}")
    if any(window.kind not in {"exponential", "cosine", "none"} for window in windows):
        raise ValueError("Window kind must be exponential, cosine, or none")
    for window in windows:
        if window.kind == "exponential" and (
            window.gamma_mev is None or window.gamma_mev <= 0
        ):
            raise ValueError("Exponential windows require gamma_mev > 0")
        if window.kind == "cosine" and (window.power is None or window.power < 1):
            raise ValueError("Cosine windows require integer power >= 1")
    for region in regions:
        if region.min_mev < 0 or region.max_mev <= region.min_mev:
            raise ValueError(f"Invalid region {region.name!r}")
    zero_padding_factor = int(analysis.get("zero_padding_factor", 8))
    baseline_points = int(analysis.get("baseline_points", 1))
    if zero_padding_factor < 1:
        raise ValueError("analysis.zero_padding_factor must be >= 1")
    if baseline_points < 1:
        raise ValueError("analysis.baseline_points must be >= 1")

    return ResponseConfig(
        config_path=path,
        config_text=text,
        run_directory=run_directory,
        input_file=input_file,
        output_directory=output_directory,
        channels=tuple(channels_list),
        windows=windows,
        regions=regions,
        energy_min_mev=float(analysis.get("energy_min_mev", 0.0)),
        energy_max_mev=float(analysis.get("energy_max_mev", 40.0)),
        zero_padding_factor=zero_padding_factor,
        baseline_points=baseline_points,
        nucleus_a=int(nucleus["A"]) if "A" in nucleus else None,
        nucleus_z=int(nucleus["Z"]) if "Z" in nucleus else None,
        trk_enhancement=float(nucleus.get("trk_enhancement", 0.0)),
        output_formats=formats,
    )
