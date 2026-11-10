"""Protect failure evidence and the separation of norm and overlap errors."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from study_orthogonality import execute, history, matched_delta, overlap_diagnostics


class OrthogonalityStudy(unittest.TestCase):
    def state(self):
        # Ten orbitals in each species; orthogonal species use the same basis.
        return dict(nucleons=(10, 10), spacing=(1, 1, 1), particles=[10, 10],
                    psi=np.concatenate([np.eye(10), np.eye(10)]).astype(complex))

    def test_off_diagonal_error_is_detected_with_normalized_orbitals(self):
        state = self.state()
        state['psi'][1] = np.cos(1e-4)*state['psi'][1] + np.sin(1e-4)*state['psi'][0]
        result = overlap_diagnostics(state, [np.eye(10), np.eye(10)])
        self.assertLess(result['norm_max_abs_error'], 1e-14)
        self.assertGreater(result['overlap_max_abs'], 9e-5)
        self.assertFalse(result['invariant_passed'])

    def test_normalization_error_is_separate_from_off_diagonal_overlap(self):
        state = self.state()
        state['psi'][0] *= 1.00001
        result = overlap_diagnostics(state, [np.eye(10), np.eye(10)])
        self.assertEqual(result['overlap_max_abs'], 0)
        self.assertGreater(result['norm_max_abs_error'], 1e-5)
        self.assertFalse(result['invariant_passed'])

    def test_matched_baseline_removes_shared_static_motion(self):
        boost = np.array([[0, 5], [2, 9], [4, 12]])
        baseline = np.array([[0, 4], [2, 7], [4, 9]])
        np.testing.assert_equal(matched_delta(boost, baseline), [0, 1, 2])

    def test_shifted_sample_times_cannot_be_subtracted(self):
        with self.assertRaisesRegex(ValueError, 'identical physical sample times'):
            matched_delta(np.array([[0, 1], [2, 2]]), np.array([[0, 1], [1, 2]]))

    def test_zero_exit_with_partial_checkpoint_is_retained_as_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'case'
            def process(*args, **kwargs):
                (path/'k0_restart.tdhf').write_bytes(b'partial checkpoint')
                return type('Result', (), dict(returncode=0))()
            with patch('study_orthogonality.subprocess.run', side_effect=process), patch(
                    'study_orthogonality.checkpoint', return_value=dict(step=1, time=0.2)):
                result = execute(Path('/unused'), path, 'nt=100', 'cpu', 1, 20, 100, [])
            self.assertEqual(result['returncode'], 0)
            self.assertFalse(result['completed'])
            self.assertFalse(result['invariant_passed'])
            self.assertIn('Incomplete checkpoint', result['failure'])

    def test_missing_endpoint_history_cannot_pass(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)/'q.res'
            np.savetxt(path, [[0, 1], [2, 2]])
            with self.assertRaisesRegex(ValueError, 'Missing exact'):
                history(path, 4)


if __name__ == '__main__':
    unittest.main()
