"""Machine-readable CSV/HDF5 writers for response products."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .models import ResponseConfig, Spectrum, SummaryRow


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return slug or "unnamed"


def _csv_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return "nan"
        if np.isposinf(value):
            return "inf"
        if np.isneginf(value):
            return "-inf"
        return f"{float(value):.12g}"
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    return value


def write_csv_outputs(
    output_directory: Path,
    spectra: Iterable[Spectrum],
    summaries: Iterable[SummaryRow],
    energy_min_mev: float,
    energy_max_mev: float,
) -> list[Path]:
    output_directory.mkdir(parents=True, exist_ok=True)
    spectra_list = list(spectra)
    written: list[Path] = []

    spectrum_path = output_directory / "spectra.csv"
    spectrum_fields = [
        "channel",
        "window",
        "energy_mev",
        "response_real",
        "response_imag",
        "signed_strength",
        "dB_dE",
        "photoabsorption_mb",
        "strength_units",
    ]
    with spectrum_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=spectrum_fields, lineterminator="\n")
        writer.writeheader()
        for spectrum in spectra_list:
            mask = (spectrum.energy_mev >= energy_min_mev) & (
                spectrum.energy_mev <= energy_max_mev
            )
            for index in np.flatnonzero(mask):
                writer.writerow(
                    {
                        "channel": spectrum.channel.name,
                        "window": spectrum.window.name,
                        "energy_mev": _csv_value(spectrum.energy_mev[index]),
                        "response_real": _csv_value(spectrum.response.real[index]),
                        "response_imag": _csv_value(spectrum.response.imag[index]),
                        "signed_strength": _csv_value(spectrum.signed_strength[index]),
                        "dB_dE": _csv_value(spectrum.b_elambda_distribution[index]),
                        "photoabsorption_mb": _csv_value(
                            spectrum.photoabsorption_mb[index]
                        ),
                        "strength_units": spectrum.channel.strength_units,
                    }
                )
    written.append(spectrum_path)

    time_path = output_directory / "time_signals.csv"
    time_fields = [
        "channel",
        "window",
        "time_fm_c",
        "raw_signal",
        "reference_signal",
        "reference_delta",
        "delta_signal",
        "window_value",
        "filtered_signal",
    ]
    with time_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=time_fields, lineterminator="\n")
        writer.writeheader()
        for spectrum in spectra_list:
            for index in range(len(spectrum.time_fm_c)):
                writer.writerow(
                    {
                        "channel": spectrum.channel.name,
                        "window": spectrum.window.name,
                        "time_fm_c": _csv_value(spectrum.time_fm_c[index]),
                        "raw_signal": _csv_value(spectrum.raw_signal[index]),
                        "reference_signal": _csv_value(spectrum.reference_signal[index]),
                        "reference_delta": _csv_value(spectrum.reference_delta[index]),
                        "delta_signal": _csv_value(spectrum.delta_signal[index]),
                        "window_value": _csv_value(spectrum.window_values[index]),
                        "filtered_signal": _csv_value(spectrum.filtered_signal[index]),
                    }
                )
    written.append(time_path)

    summary_path = output_directory / "summary.csv"
    summary_fields = [item.name for item in fields(SummaryRow)]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, lineterminator="\n")
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: _csv_value(value) for key, value in asdict(summary).items()})
    written.append(summary_path)
    return written


def write_metadata_json(output_directory: Path, metadata: dict[str, Any]) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / "metadata.json"
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def _set_attrs(target: Any, values: dict[str, object]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        if isinstance(value, Path):
            target.attrs[key] = str(value)
        elif isinstance(value, (dict, list, tuple)):
            target.attrs[key] = json.dumps(value, sort_keys=True)
        else:
            target.attrs[key] = value


def write_hdf5(
    path: Path,
    config: ResponseConfig,
    spectra: Iterable[Spectrum],
    summaries: Iterable[SummaryRow],
    metadata: dict[str, Any],
) -> Path:
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError(
            "HDF5 output requires h5py; install requirements.txt in the WSL environment"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    spectra_list = list(spectra)
    summary_list = list(summaries)
    string_type = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w", track_order=True) as handle:
        handle.attrs["schema"] = "sky3d-response-v1"
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.create_dataset("config_toml", data=config.config_text, dtype=string_type)
        channels_group = handle.create_group("channels", track_order=True)
        for channel in config.channels:
            matching = [item for item in spectra_list if item.channel.name == channel.name]
            if not matching:
                continue
            channel_group = channels_group.create_group(slugify(channel.name), track_order=True)
            first = matching[0]
            _set_attrs(channel_group, asdict(channel))
            channel_group.create_dataset("time_fm_c", data=first.time_fm_c, compression="gzip")
            channel_group.create_dataset("raw_signal", data=first.raw_signal, compression="gzip")
            channel_group.create_dataset(
                "reference_signal", data=first.reference_signal, compression="gzip"
            )
            channel_group.create_dataset(
                "reference_delta", data=first.reference_delta, compression="gzip"
            )
            channel_group.create_dataset("delta_signal", data=first.delta_signal, compression="gzip")
            windows_group = channel_group.create_group("windows", track_order=True)
            for spectrum in matching:
                group = windows_group.create_group(slugify(spectrum.window.name), track_order=True)
                _set_attrs(group, asdict(spectrum.window))
                _set_attrs(
                    group,
                    {
                        "sample_dt_fm_c": spectrum.sample_dt_fm_c,
                        "propagation_time_fm_c": spectrum.propagation_time_fm_c,
                        "rayleigh_resolution_mev": spectrum.rayleigh_resolution_mev,
                        "energy_bin_mev": spectrum.energy_bin_mev,
                        "boost_amplitude": spectrum.boost_amplitude,
                        **spectrum.metadata,
                    },
                )
                group.create_dataset("window_value", data=spectrum.window_values, compression="gzip")
                group.create_dataset("filtered_signal", data=spectrum.filtered_signal, compression="gzip")
                group.create_dataset("energy_mev", data=spectrum.energy_mev, compression="gzip")
                group.create_dataset("response_real", data=spectrum.response.real, compression="gzip")
                group.create_dataset("response_imag", data=spectrum.response.imag, compression="gzip")
                group.create_dataset(
                    "signed_strength", data=spectrum.signed_strength, compression="gzip"
                )
                group.create_dataset(
                    "dB_dE", data=spectrum.b_elambda_distribution, compression="gzip"
                )
                group.create_dataset(
                    "photoabsorption_mb", data=spectrum.photoabsorption_mb, compression="gzip"
                )

        summary_group = handle.create_group("summary", track_order=True)
        for field in fields(SummaryRow):
            values = [getattr(item, field.name) for item in summary_list]
            if not values:
                summary_group.create_dataset(field.name, data=np.array([], dtype=float))
            elif isinstance(values[0], str):
                summary_group.create_dataset(field.name, data=values, dtype=string_type)
            else:
                summary_group.create_dataset(field.name, data=values)
    return path
