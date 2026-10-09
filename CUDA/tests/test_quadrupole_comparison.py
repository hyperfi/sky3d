"""Guard matching time histories and response-baseline conventions."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_quadrupole import signal_error, transform


class QuadrupoleComparison(unittest.TestCase):
    def test_raw_offset_is_reported_while_dynamic_signal_is_compared(self):
        time = np.arange(10.)
        a = np.column_stack((time, 60 + np.sin(time), np.zeros(10)))
        b = a.copy()
        b[:, 1] += 3
        error = signal_error(a, b)
        self.assertAlmostEqual(error['raw_max_abs_fm2'], 3)
        self.assertLess(error['baseline_subtracted']['max_abs'], 1e-13)

    def test_equal_endpoints_do_not_allow_different_sample_times(self):
        a = np.array([[0, 1, 0], [1, 2, 0], [2, 3, 0]], dtype=float)
        b = a.copy()
        b[1, 0] = 0.5
        with self.assertRaises(ValueError):
            signal_error(a, b)

    def test_shared_background_drift_cancels_from_response(self):
        time = np.arange(100.) * 2
        drift = 0.001 * time
        base = np.column_stack((time, 60+drift, np.zeros(100)))
        boosted = base.copy()
        boosted[:, 1] += 0.1 * np.sin(time * 0.1)
        pure = base.copy()
        pure[:, 1] = 60 + 0.1 * np.sin(time * 0.1)
        flat = base.copy()
        flat[:, 1] = 60
        np.testing.assert_allclose(transform(boosted, base).signed_strength,
                                   transform(pure, flat).signed_strength, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
