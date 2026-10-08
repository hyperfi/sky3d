from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from sky3d_response.constants import HBARC_MEV_FM
from sky3d_response.pipeline import run_analysis


class PipelineTests(unittest.TestCase):
    def test_end_to_end_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            run.mkdir()
            eta = 5.0e-5
            time = np.arange(0.0, 2400.0 + 2.0, 2.0)
            signal = 3.0 - 2.0 * eta * 4.0 * np.sin(12.0 * time / HBARC_MEV_FM)
            (run / "for005").write_text(
                "&force name='TEST' /\n"
                "&extern ampl_ext=5D-5, L_val=1, M_val=0, only_P=1 /\n",
                encoding="utf-8",
            )
            with (run / "extfield.res").open("w", encoding="utf-8") as handle:
                handle.write("# time electric\n")
                for sample_time, sample in zip(time, signal):
                    handle.write(f"{sample_time:.6f} {sample:.15e}\n")
            config = root / "response.toml"
            config.write_text(
                "[run]\n"
                "directory='run'\n"
                "[output]\n"
                "directory='output'\n"
                "formats=['csv','hdf5','png']\n"
                "[analysis]\n"
                "energy_min_mev=0.0\nenergy_max_mev=25.0\nzero_padding_factor=4\n"
                "[nucleus]\nA=16\nZ=8\n"
                "[[windows]]\nname='gamma1'\nkind='exponential'\ngamma_mev=1.0\n"
                "[[regions]]\nname='mode'\nmin_mev=5.0\nmax_mev=20.0\n"
                "[[channels]]\nname='e1'\nfile='extfield.res'\ncolumn=1\n"
                "multipolarity=1\ntype='electric'\ndiagonal=true\nb_elambda=true\n"
                "strength_units='e^2 fm^2 / MeV'\nuse_trk_ewsr=true\n",
                encoding="utf-8",
            )
            result = run_analysis(config)
            output = root / "output"
            expected = {
                "spectra.csv",
                "time_signals.csv",
                "summary.csv",
                "response.h5",
                "metadata.json",
                "spectrum_e1.png",
                "spectra_overview.png",
            }
            self.assertTrue(expected.issubset({item.name for item in result.artifacts}))
            with (output / "summary.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertAlmostEqual(float(rows[0]["peak_energy_mev"]), 12.0, delta=0.15)
            with h5py.File(output / "response.h5", "r") as handle:
                self.assertEqual(handle.attrs["schema"], "sky3d-response-v1")
                self.assertIn("channels/e1/windows/gamma1/signed_strength", handle)
                self.assertIn("summary/m_minus1", handle)
            self.assertGreater((output / "spectrum_e1.png").stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()
