"""Readers and compact diagnostics for Sky3D unformatted ``*.tdd`` files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class DensitySnapshot:
    iteration: int
    time_fm_c: float
    spacing_fm: tuple[float, float, float]
    cell_volume_fm3: float
    x_fm: np.ndarray
    y_fm: np.ndarray
    z_fm: np.ndarray
    density_fm3: np.ndarray


def _read_record(handle) -> bytes | None:
    marker = handle.read(4)
    if not marker:
        return None
    if len(marker) != 4:
        raise ValueError("truncated Fortran record marker")
    (size,) = struct.unpack("<i", marker)
    if size < 0:
        raise ValueError("unsupported negative Fortran record size")
    payload = handle.read(size)
    trailer = handle.read(4)
    if len(payload) != size or len(trailer) != 4:
        raise ValueError("truncated Fortran record")
    (trailing_size,) = struct.unpack("<i", trailer)
    if trailing_size != size:
        raise ValueError("Fortran record markers do not agree")
    return payload


def read_tdd_density(path: Path) -> DensitySnapshot:
    """Read the total density from a little-endian Sky3D ``*.tdd`` file."""

    with path.open("rb") as handle:
        header = _read_record(handle)
        if header is None or len(header) != 24:
            raise ValueError(f"{path}: invalid density header")
        iteration, time_fm_c, nx, ny, nz = struct.unpack("<idiii", header)
        coordinates = _read_record(handle)
        if coordinates is None:
            raise ValueError(f"{path}: missing coordinate record")
        coordinate_values = np.frombuffer(coordinates, dtype="<f8")
        expected = 4 + nx + ny + nz
        if len(coordinate_values) != expected:
            raise ValueError(
                f"{path}: coordinate record has {len(coordinate_values)} values, expected {expected}"
            )
        dx, dy, dz, cell_volume = map(float, coordinate_values[:4])
        offset = 4
        x = coordinate_values[offset : offset + nx].copy()
        offset += nx
        y = coordinate_values[offset : offset + ny].copy()
        offset += ny
        z = coordinate_values[offset : offset + nz].copy()

        density = None
        while True:
            identifier = _read_record(handle)
            if identifier is None:
                break
            if len(identifier) != 18:
                raise ValueError(f"{path}: invalid field identifier record")
            raw_name, is_vector, has_isospin = struct.unpack("<10sii", identifier)
            values_record = _read_record(handle)
            if values_record is None:
                raise ValueError(f"{path}: missing values for field {raw_name!r}")
            name = raw_name.decode("ascii", errors="strict").strip().lower()
            if name != "rho":
                continue
            if is_vector:
                raise ValueError(f"{path}: Rho field unexpectedly marked as a vector")
            values = np.frombuffer(values_record, dtype="<f8")
            field_count = nx * ny * nz
            expected_count = field_count * (2 if has_isospin else 1)
            if len(values) != expected_count:
                raise ValueError(
                    f"{path}: Rho field has {len(values)} values, expected {expected_count}"
                )
            if has_isospin:
                density = values.reshape((nx, ny, nz, 2), order="F").sum(axis=3)
            else:
                density = values.reshape((nx, ny, nz), order="F").copy()

    if density is None:
        raise ValueError(f"{path}: no Rho field found")
    return DensitySnapshot(
        iteration=iteration,
        time_fm_c=time_fm_c,
        spacing_fm=(dx, dy, dz),
        cell_volume_fm3=cell_volume,
        x_fm=x,
        y_fm=y,
        z_fm=z,
        density_fm3=density,
    )


def density_diagnostics(
    snapshot: DensitySnapshot, *, shell_thickness_fm: float = 2.0
) -> dict[str, float | int]:
    """Return normalization, size, and boundary-shell diagnostics."""

    if shell_thickness_fm <= 0:
        raise ValueError("shell_thickness_fm must be positive")
    rho = snapshot.density_fm3
    dv = snapshot.cell_volume_fm3
    x, y, z = np.meshgrid(
        snapshot.x_fm, snapshot.y_fm, snapshot.z_fm, indexing="ij"
    )
    particle_number = float(np.sum(rho) * dv)
    if particle_number <= 0:
        raise ValueError("integrated density must be positive")
    radius_squared = x * x + y * y + z * z
    rms_radius = float(np.sqrt(np.sum(rho * radius_squared) * dv / particle_number))
    center = tuple(
        float(np.sum(rho * coordinate) * dv / particle_number)
        for coordinate in (x, y, z)
    )

    dx, dy, dz = snapshot.spacing_fm
    face_x = 0.5 * len(snapshot.x_fm) * dx - np.abs(x)
    face_y = 0.5 * len(snapshot.y_fm) * dy - np.abs(y)
    face_z = 0.5 * len(snapshot.z_fm) * dz - np.abs(z)
    face_distance = np.minimum(np.minimum(face_x, face_y), face_z)
    shell = face_distance <= shell_thickness_fm + 1.0e-12
    outermost = (
        (face_x <= 0.5 * dx + 1.0e-12)
        | (face_y <= 0.5 * dy + 1.0e-12)
        | (face_z <= 0.5 * dz + 1.0e-12)
    )
    shell_particles = float(np.sum(rho[shell]) * dv)

    return {
        "iteration": snapshot.iteration,
        "time_fm_c": snapshot.time_fm_c,
        "nx": len(snapshot.x_fm),
        "ny": len(snapshot.y_fm),
        "nz": len(snapshot.z_fm),
        "particle_number": particle_number,
        "rms_radius_fm": rms_radius,
        "center_x_fm": center[0],
        "center_y_fm": center[1],
        "center_z_fm": center[2],
        "shell_thickness_fm": shell_thickness_fm,
        "shell_particles": shell_particles,
        "shell_fraction": shell_particles / particle_number,
        "outermost_max_density_fm3": float(np.max(rho[outermost])),
        "outermost_mean_density_fm3": float(np.mean(rho[outermost])),
        "minimum_density_fm3": float(np.min(rho)),
        "maximum_density_fm3": float(np.max(rho)),
    }
