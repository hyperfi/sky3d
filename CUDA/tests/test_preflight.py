"""Memory-policy checks that run without CUDA or a GPU."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('preflight', Path(__file__).resolve().parents[1] / 'preflight.py')
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class MemoryPolicy(unittest.TestCase):
    def test_states_include_empty_orbitals(self):
        a = preflight.allocation_bytes([24, 24, 24], 20)
        b = preflight.allocation_bytes([24, 24, 24], 24)
        self.assertEqual(b-a, 24**3 * 2 * 4 * 16 * 9 + 4 * 52 + 54 * 40 * 8)

    def test_rectangular_grid_and_index_limit(self):
        self.assertGreater(preflight.allocation_bytes([28, 26, 24], 20), preflight.allocation_bytes([24]*3, 20))
        for grid, states in [([25]*3, 20), ([0]*3, 20), ([24]*3, 0), ([2048]*3, 208)]:
            with self.assertRaises(ValueError):
                preflight.allocation_bytes(grid, states)

    def test_pressure_and_refusal(self):
        self.assertEqual(preflight.assess(60, 100)[0], 'fits')
        self.assertEqual(preflight.assess(70, 100)[0], 'pressure')
        self.assertEqual(preflight.assess(80, 100)[0], 'pressure')
        status, message = preflight.assess(81, 100)
        self.assertEqual(status, 'insufficient')
        self.assertIn('parallel CPU', message)
        self.assertIn('refuses oversubscription', message)

    def test_no_speedup_prediction_from_capacity(self):
        self.assertIn('benchmark', preflight.assess(10, 100)[1])
        for fraction in (0, 1, float('nan')):
            with self.assertRaises(ValueError):
                preflight.assess(10, 100, fraction)

    def test_checkpoint_count_not_mass_number(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = struct.pack('<id8siii', 0, 0., b'Sly5    ', 24, 10, 10)
            (root / 'state').write_bytes(struct.pack('<i', len(header)) + header + struct.pack('<i', len(header)))
            (root / 'for005').write_text("&main nof=1 /\n&grid nx=24,ny=24,nz=24 /\n&fragments filename=1*'state' /")
            self.assertEqual(preflight.input_case(root / 'for005'), ([24, 24, 24], 24))
            (root / 'for005').write_text("&main nof=2 /\n&grid nx=24,ny=24,nz=24 /\n&fragments filename='state' /")
            with self.assertRaises(ValueError):
                preflight.input_case(root / 'for005')


if __name__ == '__main__':
    unittest.main()
