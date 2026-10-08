from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sky3d_response.density import density_diagnostics, read_tdd_density


def write_record(handle, payload: bytes) -> None:
    marker = struct.pack("<i", len(payload))
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


class DensityTests(unittest.TestCase):
    def test_read_density_and_boundary_diagnostics(self) -> None:
        nx = ny = nz = 4
        axis = np.asarray([-1.5, -0.5, 0.5, 1.5], dtype="<f8")
        density = np.ones((nx, ny, nz), dtype="<f8") / 8.0
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "000020.tdd"
            with path.open("wb") as handle:
                write_record(handle, struct.pack("<idiii", 20, 4.0, nx, ny, nz))
                coordinates = np.concatenate(([1.0, 1.0, 1.0, 1.0], axis, axis, axis))
                write_record(handle, np.asarray(coordinates, dtype="<f8").tobytes())
                write_record(handle, struct.pack("<10sii", b"Rho       ", 0, 0))
                write_record(handle, density.tobytes(order="F"))
            snapshot = read_tdd_density(path)
            result = density_diagnostics(snapshot, shell_thickness_fm=1.0)
        self.assertEqual(snapshot.iteration, 20)
        self.assertEqual(snapshot.time_fm_c, 4.0)
        self.assertAlmostEqual(result["particle_number"], 8.0)
        self.assertAlmostEqual(result["shell_particles"], 7.0)
        self.assertAlmostEqual(result["rms_radius_fm"], np.sqrt(3.75))
        self.assertAlmostEqual(result["outermost_max_density_fm3"], 0.125)


if __name__ == "__main__":
    unittest.main()
