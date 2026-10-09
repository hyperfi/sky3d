"""Guard against comparing mismatched physical times or different species."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from validate_extended import gram, integration_difference


class ExtendedValidation(unittest.TestCase):
    def test_overlaps_are_per_species_with_cell_volume(self):
        # Neutron/proton states can share spatial/spin wavefunctions.
        state = dict(nucleons=(2, 2), spacing=(0.5, 0.5, 0.5),
                     psi=np.sqrt(8) * np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=complex))
        for matrix in gram(state):
            np.testing.assert_allclose(matrix, np.eye(2), atol=1e-14)

    def compare_fixture(self, second_times):
        with tempfile.TemporaryDirectory() as temporary:
            paths = [Path(temporary) / name for name in ('coarse', 'fine')]
            for path, times in zip(paths, ([0, 1, 2], second_times)):
                path.mkdir()
                for name in ('energies.res', 'quadrupoles.res', 'monopoles.res'):
                    values = np.zeros((3, 8))
                    values[:, 0] = times
                    np.savetxt(path / name, values)
            with patch('validate_extended.checkpoint', side_effect=[
                dict(grid=(2, 2, 2), time=2, step=20, psi=np.ones(2)),
                dict(grid=(2, 2, 2), time=2, step=40, psi=np.ones(2))]):
                return integration_difference(*paths)

    def test_different_step_counts_can_share_physical_times(self):
        self.assertEqual(self.compare_fixture([0, 1, 2])['wavefunction']['max_abs'], 0)

    def test_same_endpoint_does_not_justify_mismatched_history(self):
        with self.assertRaisesRegex(ValueError, 'matched physical output times'):
            self.compare_fixture([0, 0.5, 2])


if __name__ == '__main__':
    unittest.main()
