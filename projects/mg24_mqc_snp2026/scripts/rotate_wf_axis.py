#!/usr/bin/env python3
"""Rotate a sequential Sky3D wavefunction from a long x axis to long z.

The active rotation is R_y(-pi/2): x -> z.  On an equal x/z Cartesian
grid this is an exact index permutation.  The two-component spinors are
rotated by the matching SU(2) matrix; no interpolation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np


def read_record(handle) -> bytes | None:
    marker = handle.read(4)
    if not marker:
        return None
    if len(marker) != 4:
        raise ValueError("truncated leading Fortran record marker")
    (size,) = struct.unpack("<i", marker)
    payload = handle.read(size)
    trailer = handle.read(4)
    if len(payload) != size or len(trailer) != 4:
        raise ValueError("truncated Fortran record")
    (end_size,) = struct.unpack("<i", trailer)
    if end_size != size:
        raise ValueError(f"record-marker mismatch: {size} != {end_size}")
    return payload


def write_record(handle, payload: bytes) -> None:
    marker = struct.pack("<i", len(payload))
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rotate(input_path: Path, output_path: Path) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")

    with input_path.open("rb") as source:
        records: list[bytes] = []
        while (record := read_record(source)) is not None:
            records.append(record)

    if len(records) < 6:
        raise ValueError("file has too few records to be a Sky3D wavefunction")

    nx, ny, nz = struct.unpack_from("<3i", records[1], 0)
    if nx != nz:
        raise ValueError(f"x/z rotation requires nx == nz, got {nx} and {nz}")

    coordinates = np.frombuffer(records[2], dtype="<f8")
    if coordinates.size != nx + ny + nz:
        raise ValueError("unexpected coordinate-record size")
    x = coordinates[:nx]
    z = coordinates[nx + ny :]
    if not np.allclose(x, z, rtol=0.0, atol=1.0e-13):
        raise ValueError("x and z coordinate grids differ")

    wave_record_bytes = nx * ny * nz * 2 * np.dtype("<c16").itemsize
    wave_indices = [index for index, record in enumerate(records) if len(record) == wave_record_bytes]
    if not wave_indices:
        raise ValueError("no wavefunction records found")

    norm_errors: list[float] = []
    inv_sqrt_two = 1.0 / np.sqrt(2.0)
    for index in wave_indices:
        psi = np.frombuffer(records[index], dtype="<c16").reshape(
            (nx, ny, nz, 2), order="F"
        )
        # new(x,y,z) = old(z,y,-x) for R_y(-pi/2)
        spatial = np.transpose(psi, (2, 1, 0, 3))[::-1, :, :, :]
        rotated = np.empty_like(spatial)
        rotated[..., 0] = inv_sqrt_two * (spatial[..., 0] + spatial[..., 1])
        rotated[..., 1] = inv_sqrt_two * (-spatial[..., 0] + spatial[..., 1])
        before = float(np.vdot(psi, psi).real)
        after = float(np.vdot(rotated, rotated).real)
        norm_errors.append(abs(after - before) / before)
        records[index] = np.asfortranarray(rotated).tobytes(order="F")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as destination:
        for record in records:
            write_record(destination, record)

    return {
        "input": str(input_path.resolve()),
        "output": str(output_path.resolve()),
        "input_sha256": sha256(input_path),
        "output_sha256": sha256(output_path),
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "wavefunctions_rotated": len(wave_indices),
        "rotation": "active R_y(-pi/2): x -> z",
        "interpolation": False,
        "max_relative_discrete_norm_error": max(norm_errors),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()

    metadata = rotate(args.input, args.output)
    rendered = json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    if args.metadata:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()

