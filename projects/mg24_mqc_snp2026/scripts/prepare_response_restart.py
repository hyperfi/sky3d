#!/usr/bin/env python3
"""Stage a Sky3D response restart without duplicating post-checkpoint data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path


TOLERANCE = 1.0e-6


def restart_header(path: Path) -> tuple[int, float, str]:
    """Read iter, time, and force name from Sky3D's first Fortran record."""
    with path.open("rb") as stream:
        marker_raw = stream.read(4)
        if len(marker_raw) != 4:
            raise ValueError(f"Truncated restart file: {path}")
        (record_bytes,) = struct.unpack("<i", marker_raw)
        if record_bytes < 20 or record_bytes > 1_000_000:
            raise ValueError(f"Unexpected first-record size {record_bytes} in {path}")
        payload = stream.read(record_bytes)
        closing_raw = stream.read(4)
    if len(payload) != record_bytes or len(closing_raw) != 4:
        raise ValueError(f"Truncated first record in {path}")
    (closing_bytes,) = struct.unpack("<i", closing_raw)
    if closing_bytes != record_bytes:
        raise ValueError(f"Mismatched Fortran record markers in {path}")
    iteration, time = struct.unpack_from("<id", payload, 0)
    force = payload[12:20].decode("ascii", errors="replace").strip()
    return iteration, time, force


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truncate_protocol(
    source: Path,
    target: Path,
    checkpoint_iteration: int,
    checkpoint_time: float,
) -> dict[str, object]:
    lines = source.read_text(encoding="utf-8", errors="strict").splitlines(keepends=True)
    header = next((line.strip().lower() for line in lines if line.lstrip().startswith("#")), "")
    numeric_values = []
    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            continue
        try:
            numeric_values.append(float(stripped.split()[0]))
        except (ValueError, IndexError):
            continue
    # Some legacy protocols (notably dipoles.res) label the first column
    # "Iter" while actually writing physical time. Select the checkpoint scale
    # that best matches the observed endpoint instead of trusting that label.
    if numeric_values and checkpoint_time and checkpoint_iteration:
        endpoint = numeric_values[-1]
        time_score = abs(endpoint / checkpoint_time - 1.0)
        iteration_score = abs(endpoint / checkpoint_iteration - 1.0)
        domain = "time_fm_c" if time_score <= iteration_score else "iteration"
    else:
        domain = "iteration" if re.match(r"^#\s*iter\b", header) else "time_fm_c"
    cutoff = float(checkpoint_iteration) if domain == "iteration" else checkpoint_time
    tolerance = 0.5 if domain == "iteration" else TOLERANCE
    kept: list[str] = []
    kept_rows = 0
    dropped_rows = 0
    original_endpoint: float | None = None
    kept_endpoint: float | None = None

    for line in lines:
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            kept.append(line)
            continue
        try:
            time = float(stripped.split()[0])
        except (ValueError, IndexError):
            kept.append(line)
            continue
        original_endpoint = time
        # The restarted calculation writes its own checkpoint-time row. Keep only
        # earlier samples so the concatenated protocol is strictly increasing.
        if time < cutoff - tolerance:
            kept.append(line)
            kept_rows += 1
            kept_endpoint = time
        else:
            dropped_rows += 1

    target.write_text("".join(kept), encoding="utf-8")
    shutil.copystat(source, target)
    return {
        "file": source.name,
        "first_column": domain,
        "checkpoint_value": cutoff,
        "kept_numeric_rows": kept_rows,
        "dropped_numeric_rows": dropped_rows,
        "kept_endpoint": kept_endpoint,
        "original_endpoint": original_endpoint,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--restart-file", required=True, type=Path)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    run_dir = args.run_dir.resolve()
    restart_file = args.restart_file.resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Missing source directory: {source_dir}")
    if not restart_file.is_file():
        raise SystemExit(f"Missing restart file: {restart_file}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise SystemExit(f"Refusing to stage into non-empty directory: {run_dir}")

    iteration, time, force = restart_header(restart_file)
    protocol_records = []
    for source in sorted(source_dir.glob("*.res")):
        protocol_records.append(
            truncate_protocol(source, run_dir / source.name, iteration, time)
        )

    density_files = []
    for source in sorted(source_dir.glob("*.tdd")):
        match = re.fullmatch(r"(\d{6})\.tdd", source.name)
        if match and int(match.group(1)) <= iteration:
            shutil.copy2(source, run_dir / source.name)
            density_files.append(source.name)

    restart_target = run_dir / "restart.tdhf"
    shutil.copy2(restart_file, restart_target)
    metadata = {
        "source_run": str(source_dir),
        "source_restart": str(restart_file),
        "source_restart_sha256": sha256(restart_file),
        "checkpoint_iteration": iteration,
        "checkpoint_time_fm_c": time,
        "force": force,
        "protocols": protocol_records,
        "density_files_copied": density_files,
        "policy": (
            "Keep protocol rows strictly before checkpoint; restart writes checkpoint row. "
            "Infer time versus iteration scale from the endpoint because legacy dipoles.res "
            "is mislabeled as Iter."
        ),
    }
    (run_dir / "restart_provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
