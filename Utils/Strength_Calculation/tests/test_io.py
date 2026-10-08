from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sky3d_response.io import parse_sky3d_input, read_response_file


class InputOutputTests(unittest.TestCase):
    def test_parse_fortran_scalars_and_d_exponent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "for005"
            path.write_text(
                "&force name='SV-bas', pairing='NONE' /\n"
                "&extern ampl_ext=5D-5, L_val=2, M_val=0, "
                "textfield_periodic=F /\n"
                "&fragments filename=1*'../Static/O16', fix_boost=T /\n",
                encoding="utf-8",
            )
            parsed = parse_sky3d_input(path)
        self.assertEqual(parsed["force"]["name"], "SV-bas")
        self.assertAlmostEqual(parsed["extern"]["ampl_ext"], 5.0e-5)
        self.assertEqual(parsed["extern"]["l_val"], 2)
        self.assertFalse(parsed["extern"]["textfield_periodic"])
        self.assertEqual(parsed["fragments"]["filename"], "1*'../Static/O16'")

    def test_read_response_column_and_fortran_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "quadrupoles.res"
            path.write_text(
                "# time IS IV\n0.0 1.0D+0 -2.0\n2.0 1.5D+0 -2.5\n4.0 2.0D+0 -3.0\n",
                encoding="utf-8",
            )
            time, signal, header = read_response_file(path, 2)
        np.testing.assert_allclose(time, [0.0, 2.0, 4.0])
        np.testing.assert_allclose(signal, [-2.0, -2.5, -3.0])
        self.assertEqual(header, "time IS IV")

    def test_read_response_deduplicates_restart_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "monopoles.res"
            path.write_text(
                "# time IS IV\n"
                "0.0 1.0 2.0\n"
                "2.0 1.5 2.5\n"
                "2.0 1.5 2.5\n"
                "4.0 2.0 3.0\n",
                encoding="utf-8",
            )
            time, signal, _ = read_response_file(path, 1)
        np.testing.assert_allclose(time, [0.0, 2.0, 4.0])
        np.testing.assert_allclose(signal, [1.0, 1.5, 2.0])

    def test_read_response_rejects_inconsistent_restart_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "monopoles.res"
            path.write_text(
                "# time IS IV\n"
                "0.0 1.0 2.0\n"
                "2.0 1.5 2.5\n"
                "2.0 1.6 2.5\n"
                "4.0 2.0 3.0\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate time"):
                read_response_file(path, 1)


if __name__ == "__main__":
    unittest.main()
