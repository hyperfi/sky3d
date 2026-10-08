from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from sky3d_response.analysis import analyze_signal, calculate_summary, trk_ewsr
from sky3d_response.constants import E1_PHOTOABSORPTION_MB, E2_MEV_FM, HBARC_MEV_FM
from sky3d_response.models import ChannelSpec, RegionSpec, WindowSpec


class AnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time = np.arange(0.0, 6000.0 + 1.0, 1.0)
        self.energy = 15.0
        self.transition_strength = 7.5
        self.eta = 1.0e-4
        self.signal = 4.0 - 2.0 * self.eta * self.transition_strength * np.sin(
            self.energy * self.time / HBARC_MEV_FM
        )
        self.channel = ChannelSpec(
            name="analytic_e1",
            file=Path("synthetic.res"),
            column=1,
            multipolarity=1,
            channel_type="electric",
            diagonal=True,
            strength_units="e^2 fm^2 / MeV",
            b_elambda=True,
            ewsr_reference=self.energy * self.transition_strength,
            ewsr_label="analytic",
        )
        self.window = WindowSpec(name="gamma_0p8", kind="exponential", gamma_mev=0.8)

    def test_single_mode_recovers_strength_peak_and_width(self) -> None:
        spectrum = analyze_signal(
            self.time,
            self.signal,
            self.channel,
            self.window,
            boost_amplitude=self.eta,
            zero_padding_factor=4,
        )
        summary = calculate_summary(spectrum, RegionSpec("mode", 5.0, 25.0))
        self.assertAlmostEqual(summary.peak_energy_mev, self.energy, delta=0.08)
        # A finite 5--25 MeV interval excludes the Lorentzian tails.
        self.assertAlmostEqual(summary.m0, self.transition_strength, delta=0.25)
        self.assertAlmostEqual(summary.centroid_m1_m0_mev, self.energy, delta=0.25)
        self.assertAlmostEqual(summary.observed_fwhm_mev, 0.8, delta=0.10)
        self.assertEqual(summary.artificial_smoothing_fwhm_mev, 0.8)
        self.assertAlmostEqual(summary.ewsr_exhaustion_percent, 100.0, delta=3.0)
        expected_alpha_d = (8.0 * np.pi / 9.0) * E2_MEV_FM * (
            self.transition_strength / self.energy
        )
        self.assertAlmostEqual(summary.alpha_d_fm3, expected_alpha_d, delta=0.12)
        peak = int(np.argmax(spectrum.b_elambda_distribution))
        expected_sigma = (
            E1_PHOTOABSORPTION_MB
            * spectrum.energy_mev[peak]
            * spectrum.b_elambda_distribution[peak]
        )
        self.assertAlmostEqual(spectrum.photoabsorption_mb[peak], expected_sigma)

    def test_cross_response_retains_negative_sign(self) -> None:
        cross_channel = ChannelSpec(
            name="cross",
            file=Path("synthetic.res"),
            column=1,
            multipolarity=2,
            channel_type="cross",
            diagonal=False,
            strength_units="fm^4 / MeV",
        )
        negative_product_signal = 2.0 - 2.0 * self.eta * (-3.0) * np.sin(
            self.energy * self.time / HBARC_MEV_FM
        )
        spectrum = analyze_signal(
            self.time,
            negative_product_signal,
            cross_channel,
            self.window,
            boost_amplitude=self.eta,
        )
        summary = calculate_summary(spectrum, RegionSpec("mode", 5.0, 25.0))
        self.assertLess(float(np.min(spectrum.signed_strength)), -1.0)
        self.assertTrue(np.isnan(summary.centroid_m1_m0_mev))
        self.assertAlmostEqual(summary.peak_energy_mev, self.energy, delta=0.08)

    def test_reference_subtraction_removes_unboosted_background(self) -> None:
        background = 0.002 * np.sin(7.0 * self.time / HBARC_MEV_FM)
        reference = 9.0 + background
        contaminated = self.signal + background
        spectrum = analyze_signal(
            self.time,
            contaminated,
            self.channel,
            self.window,
            boost_amplitude=self.eta,
            reference_signal=reference,
            zero_padding_factor=4,
        )
        clean = analyze_signal(
            self.time,
            self.signal,
            self.channel,
            self.window,
            boost_amplitude=self.eta,
            zero_padding_factor=4,
        )
        np.testing.assert_allclose(spectrum.delta_signal, clean.delta_signal, atol=2.0e-15)
        np.testing.assert_allclose(spectrum.response, clean.response, atol=1.0e-9)
        self.assertTrue(spectrum.metadata["reference_subtracted"])
        self.assertAlmostEqual(spectrum.metadata["reference_baseline"], 9.0)

    def test_trk_reference(self) -> None:
        expected = (9.0 / (4.0 * np.pi)) * 20.73553 * 4.0
        self.assertAlmostEqual(trk_ewsr(16, 8), expected)


if __name__ == "__main__":
    unittest.main()
