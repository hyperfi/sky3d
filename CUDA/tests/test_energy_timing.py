"""Check the isolated solver against analytic point-charge potentials."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check_energy_timing import isolated_potential


class CoulombTiming(unittest.TestCase):
    def test_one_unit_charge_has_self_and_neighbor_potentials(self):
        spacing = (0.5, 0.75, 1.0)
        rho = np.zeros((4, 6, 8))
        rho[1, 2, 3] = 1 / np.prod(spacing)
        potential = isolated_potential(rho, spacing)
        self.assertAlmostEqual(potential[1, 2, 3], 1.43989 * 2.84 * 3 / sum(spacing), places=12)
        self.assertAlmostEqual(potential[2, 2, 3], 1.43989 / spacing[0], places=12)
        self.assertAlmostEqual(potential[1, 3, 4], 1.43989 / np.hypot(spacing[1], spacing[2]), places=12)

    def test_isolated_field_does_not_wrap_physical_grid_edges(self):
        rho = np.zeros((4, 6, 8))
        rho[0, 0, 0] = 1
        potential = isolated_potential(rho, (1, 1, 1))
        self.assertAlmostEqual(potential[3, 0, 0], 1.43989 / 3, places=12)


if __name__ == '__main__':
    unittest.main()
