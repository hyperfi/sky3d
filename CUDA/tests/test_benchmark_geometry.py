"""Check that fine-grid jobs preserve geometry and explicit integration settings."""
import re
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import input_text, check_observable, difference


class InputGeometry(unittest.TestCase):
    def test_fine_and_padded_grid_keep_spacing_and_timestep(self):
        for grid in ((40, 40, 40), (44, 42, 40)):
            text = input_text(100, grid, validate=True, spacing=(0.6,)*3, dt=0.1)
            for key, value in zip(('nx', 'ny', 'nz', 'dx', 'dy', 'dz', 'dt'), (*grid, 0.6, 0.6, 0.6, 0.1)):
                actual = float(re.search(rf'\b{key}=([\d.e+-]+)', text).group(1))
                self.assertEqual(actual, value)
            self.assertIn('nt=100', text)
            self.assertIn('mxpact=4', text)
            self.assertIn('../initial.tdhf', text)

    def test_original_default_case_is_preserved(self):
        text = input_text(500)
        self.assertIn('nx=24, ny=24, nz=24', text)
        self.assertIn('dx=1, dy=1, dz=1', text)
        self.assertEqual(float(re.search(r'\bdt=([\d.e+-]+)', text).group(1)), 0.2)

    def test_resetcm_works_with_explicit_order(self):
        text = input_text(100, resetcm=True, mxpact=6, dt=0.1)
        self.assertIn('mxpact=6, mrescm=1,', text)


class EmptyCoulombOutput(unittest.TestCase):
    def test_disabled_coulomb_has_matching_empty_records(self):
        self.assertEqual(difference(np.empty(0), np.empty(0)), dict(relative_l2=0.,max_abs=0.))
        with self.assertRaises(ValueError):
            difference(np.empty(0), np.ones(1))


class FlowComparison(unittest.TestCase):
    def test_override_applies_only_at_time_zero(self):
        reference = np.zeros((2, 8))
        reference[:, 0] = [0, 1]
        actual = reference.copy()
        actual[0, 6] = 8e-6
        with self.assertRaises(AssertionError):
            check_observable('energies.res', actual, reference)
        check_observable('energies.res', actual, reference, initial_flow_atol=1e-5)
        actual[1, 6] = 8e-6
        with self.assertRaises(AssertionError):
            check_observable('energies.res', actual, reference, initial_flow_atol=1e-5)

    def test_override_keeps_total_energy_tolerance(self):
        reference = np.zeros((1, 8))
        actual = reference.copy()
        actual[0, 3] = 1e-6
        with self.assertRaises(AssertionError):
            check_observable('energies.res', actual, reference, initial_flow_atol=1e-5)


if __name__ == '__main__':
    unittest.main()
