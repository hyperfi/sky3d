"""Linear-response transforms and derived nuclear observables."""

from __future__ import annotations

from dataclasses import replace
from math import pi

import numpy as np

from .constants import (
    E1_PHOTOABSORPTION_MB,
    E2_MEV_FM,
    HBAR2_OVER_2M_NUCLEON_MEV_FM2,
    HBARC_MEV_FM,
)
from .models import ChannelSpec, RegionSpec, Spectrum, SummaryRow, WindowSpec


def make_window(time_fm_c: np.ndarray, spec: WindowSpec) -> np.ndarray:
    """Return a causal time window.

    ``exponential`` uses exp[-Gamma*t/(2*hbar)], so ``gamma_mev`` is the
    exact artificial Lorentzian FWHM. A cosine window has no unique
    Lorentzian smoothing width and is therefore reported separately.
    """

    time = np.asarray(time_fm_c, dtype=float) - float(time_fm_c[0])
    if spec.kind == "exponential":
        if spec.gamma_mev is None or spec.gamma_mev <= 0:
            raise ValueError("Exponential window requires gamma_mev > 0")
        return np.exp(-spec.gamma_mev * time / (2.0 * HBARC_MEV_FM))
    if spec.kind == "cosine":
        if spec.power is None or spec.power < 1:
            raise ValueError("Cosine window requires power >= 1")
        if time[-1] <= 0:
            raise ValueError("Time interval must be positive")
        return np.cos(0.5 * pi * time / time[-1]) ** spec.power
    if spec.kind == "none":
        return np.ones_like(time)
    raise ValueError(f"Unknown window kind {spec.kind!r}")


def _validate_time(time: np.ndarray) -> float:
    if time.ndim != 1 or len(time) < 3:
        raise ValueError("Time must be a one-dimensional array with at least three samples")
    differences = np.diff(time)
    if np.any(differences <= 0):
        raise ValueError("Time samples must increase strictly")
    dt = float(np.median(differences))
    tolerance = max(1.0e-10, abs(dt) * 1.0e-7)
    if not np.allclose(differences, dt, rtol=1.0e-7, atol=tolerance):
        maximum = float(np.max(np.abs(differences - dt)))
        raise ValueError(
            f"Response samples are not equidistant: median dt={dt:g}, max deviation={maximum:g} fm/c"
        )
    return dt


def analyze_signal(
    time_fm_c: np.ndarray,
    signal: np.ndarray,
    channel: ChannelSpec,
    window: WindowSpec,
    *,
    boost_amplitude: float,
    zero_padding_factor: int = 8,
    baseline_points: int = 1,
    reference_signal: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> Spectrum:
    """Transform a Sky3D time signal into a signed response density.

    Sky3D applies ``exp(-i*eta*F)``. With NumPy's negative-exponent FFT,
    the diagonal strength is therefore

        S(E) = Im[ integral dt exp(-iEt/hbar) delta<F>(t) ]
               / (pi * eta * hbar).

    ``boost_phase_sign`` generalizes this to ``exp(i*s*eta*F)``. The time
    integration factor ``dt`` is explicit and zero padding only interpolates
    the spectrum; it does not improve the Rayleigh resolution.
    """

    time = np.asarray(time_fm_c, dtype=float)
    values = np.asarray(signal, dtype=float)
    if values.shape != time.shape:
        raise ValueError("Signal and time arrays must have the same shape")
    dt = _validate_time(time)
    if not np.all(np.isfinite(values)):
        raise ValueError("Signal contains non-finite values")
    if boost_amplitude == 0 or not np.isfinite(boost_amplitude):
        raise ValueError("Boost amplitude must be finite and non-zero")
    if zero_padding_factor < 1:
        raise ValueError("zero_padding_factor must be >= 1")
    if baseline_points < 1 or baseline_points >= len(values):
        raise ValueError("baseline_points must be between 1 and number of samples - 1")

    time_relative = time - time[0]
    baseline = float(np.mean(values[:baseline_points]))
    if reference_signal is None:
        reference_values = np.full_like(values, np.nan)
        reference_baseline = float("nan")
        reference_delta = np.zeros_like(values)
    else:
        reference_values = np.asarray(reference_signal, dtype=float)
        if reference_values.shape != time.shape:
            raise ValueError("Reference signal and time arrays must have the same shape")
        if not np.all(np.isfinite(reference_values)):
            raise ValueError("Reference signal contains non-finite values")
        reference_baseline = float(np.mean(reference_values[:baseline_points]))
        reference_delta = (
            (reference_values - reference_baseline)
            * channel.reference_scale
            * channel.observable_scale
        )
    delta = (values - baseline) * channel.observable_scale - reference_delta
    window_values = make_window(time_relative, window)
    filtered = delta * window_values
    n_fft = len(time) * zero_padding_factor
    integral = np.fft.rfft(filtered, n=n_fft) * dt
    energy = 2.0 * pi * HBARC_MEV_FM * np.fft.rfftfreq(n_fft, d=dt)
    response = (
        -channel.boost_phase_sign
        * channel.boost_operator_scale
        * integral
        / (boost_amplitude * HBARC_MEV_FM)
    )
    signed_strength = response.imag / pi
    if channel.b_elambda:
        b_distribution = signed_strength.copy()
    else:
        b_distribution = np.full_like(signed_strength, np.nan)
    if channel.b_elambda and channel.multipolarity == 1:
        photoabsorption = E1_PHOTOABSORPTION_MB * energy * b_distribution
    else:
        photoabsorption = np.full_like(signed_strength, np.nan)

    propagation_time = float(time_relative[-1])
    rayleigh = 2.0 * pi * HBARC_MEV_FM / propagation_time
    energy_bin = float(energy[1] - energy[0])
    details = dict(metadata or {})
    details.update(
        {
            "baseline": baseline,
            "baseline_points": baseline_points,
            "reference_subtracted": reference_signal is not None,
            "reference_baseline": reference_baseline,
            "reference_scale": channel.reference_scale,
            "n_samples": len(time),
            "n_fft": n_fft,
            "zero_padding_factor": zero_padding_factor,
            "transform_kernel": "exp(-i E t / hbar)",
            "sky3d_boost": f"exp(i * {channel.boost_phase_sign:+d} * eta * F)",
            "strength_definition": "Im(response)/pi; signed for non-diagonal channels",
        }
    )
    return Spectrum(
        channel=replace(channel, boost_amplitude=boost_amplitude),
        window=window,
        time_fm_c=time,
        raw_signal=values,
        reference_signal=reference_values,
        reference_delta=reference_delta,
        delta_signal=delta,
        window_values=window_values,
        filtered_signal=filtered,
        energy_mev=energy,
        response=response,
        signed_strength=signed_strength,
        b_elambda_distribution=b_distribution,
        photoabsorption_mb=photoabsorption,
        sample_dt_fm_c=dt,
        propagation_time_fm_c=propagation_time,
        rayleigh_resolution_mev=rayleigh,
        energy_bin_mev=energy_bin,
        boost_amplitude=boost_amplitude,
        metadata=details,
    )


def _bounded_arrays(
    energy: np.ndarray, values: np.ndarray, minimum: float, maximum: float
) -> tuple[np.ndarray, np.ndarray]:
    if maximum <= minimum:
        raise ValueError("Integration maximum must exceed minimum")
    lower = max(float(minimum), float(energy[0]))
    upper = min(float(maximum), float(energy[-1]))
    if upper <= lower:
        return np.array([], dtype=float), np.array([], dtype=float)
    mask = (energy > lower) & (energy < upper)
    bounded_energy = np.concatenate(([lower], energy[mask], [upper]))
    bounded_values = np.concatenate(
        ([np.interp(lower, energy, values)], values[mask], [np.interp(upper, energy, values)])
    )
    return bounded_energy, bounded_values


def energy_moment(
    energy_mev: np.ndarray,
    strength: np.ndarray,
    order: int,
    minimum_mev: float,
    maximum_mev: float,
) -> float:
    lower = minimum_mev
    if order < 0:
        positive = energy_mev[energy_mev > 0]
        if not len(positive):
            return float("nan")
        lower = max(lower, float(positive[0]))
    energy, values = _bounded_arrays(energy_mev, strength, lower, maximum_mev)
    if len(energy) < 2:
        return float("nan")
    return float(np.trapz(values * energy**order, energy))


def _fwhm(energy: np.ndarray, strength: np.ndarray) -> tuple[float, float]:
    if len(energy) < 3 or not np.any(np.isfinite(strength)):
        return float("nan"), float("nan")
    peak_index = int(np.nanargmax(strength))
    peak_value = float(strength[peak_index])
    peak_energy = float(energy[peak_index])
    if peak_value <= 0:
        return peak_energy, float("nan")
    half = 0.5 * peak_value
    left_candidates = np.flatnonzero(strength[:peak_index] <= half)
    right_candidates = np.flatnonzero(strength[peak_index + 1 :] <= half)
    if not len(left_candidates) or not len(right_candidates):
        return peak_energy, float("nan")
    left_low = int(left_candidates[-1])
    left_high = left_low + 1
    right_high = peak_index + 1 + int(right_candidates[0])
    right_low = right_high - 1

    def crossing(i0: int, i1: int) -> float:
        y0, y1 = float(strength[i0]), float(strength[i1])
        if y1 == y0:
            return float(energy[i0])
        fraction = (half - y0) / (y1 - y0)
        return float(energy[i0] + fraction * (energy[i1] - energy[i0]))

    return peak_energy, crossing(right_low, right_high) - crossing(left_low, left_high)


def trk_ewsr(nucleus_a: int, nucleus_z: int, enhancement: float = 0.0) -> float:
    """Return the E1 Thomas-Reiche-Kuhn EWSR in MeV e^2 fm^2."""

    if nucleus_a <= 0 or nucleus_z <= 0 or nucleus_z >= nucleus_a:
        raise ValueError("TRK EWSR requires 0 < Z < A")
    neutron_number = nucleus_a - nucleus_z
    return float(
        (9.0 / (4.0 * pi))
        * HBAR2_OVER_2M_NUCLEON_MEV_FM2
        * neutron_number
        * nucleus_z
        / nucleus_a
        * (1.0 + enhancement)
    )


def calculate_summary(
    spectrum: Spectrum,
    region: RegionSpec,
    *,
    nucleus_a: int | None = None,
    nucleus_z: int | None = None,
    trk_enhancement: float = 0.0,
) -> SummaryRow:
    energy, strength = _bounded_arrays(
        spectrum.energy_mev,
        spectrum.signed_strength,
        region.min_mev,
        region.max_mev,
    )
    m_minus1 = energy_moment(
        spectrum.energy_mev,
        spectrum.signed_strength,
        -1,
        region.min_mev,
        region.max_mev,
    )
    m0 = energy_moment(
        spectrum.energy_mev,
        spectrum.signed_strength,
        0,
        region.min_mev,
        region.max_mev,
    )
    m1 = energy_moment(
        spectrum.energy_mev,
        spectrum.signed_strength,
        1,
        region.min_mev,
        region.max_mev,
    )
    centroid = m1 / m0 if spectrum.channel.diagonal and np.isfinite(m0) and m0 > 0 else float("nan")
    if spectrum.channel.diagonal:
        peak, observed_fwhm = _fwhm(energy, strength)
    else:
        peak = float(energy[int(np.argmax(np.abs(strength)))]) if len(energy) else float("nan")
        observed_fwhm = float("nan")

    artificial_fwhm = float("nan")
    intrinsic_fwhm = float("nan")
    if spectrum.window.kind == "exponential":
        artificial_fwhm = float(spectrum.window.gamma_mev)
        if np.isfinite(observed_fwhm):
            intrinsic_fwhm = max(0.0, observed_fwhm - artificial_fwhm)
    elif spectrum.window.kind == "none":
        artificial_fwhm = 0.0
        intrinsic_fwhm = observed_fwhm

    alpha_d = float("nan")
    if spectrum.channel.b_elambda and spectrum.channel.multipolarity == 1:
        alpha_d = (8.0 * pi / 9.0) * E2_MEV_FM * m_minus1

    ewsr_reference = spectrum.channel.ewsr_reference
    ewsr_label = spectrum.channel.ewsr_label or ""
    if spectrum.channel.use_trk_ewsr:
        if not spectrum.channel.b_elambda or spectrum.channel.multipolarity != 1:
            raise ValueError("TRK EWSR is only available for declared electric E1 B(E1) channels")
        if nucleus_a is None or nucleus_z is None:
            raise ValueError("TRK EWSR requires [nucleus] A and Z in the configuration")
        ewsr_reference = trk_ewsr(nucleus_a, nucleus_z, trk_enhancement)
        ewsr_label = f"TRK*(1+{trk_enhancement:g})"
    ewsr_reference_value = (
        float(ewsr_reference) if ewsr_reference is not None else float("nan")
    )
    exhaustion = (
        100.0 * m1 / ewsr_reference_value
        if spectrum.channel.diagonal
        and np.isfinite(ewsr_reference_value)
        and ewsr_reference_value > 0
        else float("nan")
    )

    absolute_area = float(np.trapz(np.abs(strength), energy)) if len(energy) else 0.0
    negative_area = (
        float(np.trapz(np.clip(-strength, 0.0, None), energy)) if len(energy) else 0.0
    )
    negative_fraction = negative_area / absolute_area if absolute_area > 0 else float("nan")
    return SummaryRow(
        channel=spectrum.channel.name,
        window=spectrum.window.name,
        region=region.name,
        energy_min_mev=region.min_mev,
        energy_max_mev=region.max_mev,
        m_minus1=m_minus1,
        m0=m0,
        m1=m1,
        centroid_m1_m0_mev=centroid,
        peak_energy_mev=peak,
        observed_fwhm_mev=observed_fwhm,
        artificial_smoothing_fwhm_mev=artificial_fwhm,
        intrinsic_fwhm_lorentzian_estimate_mev=intrinsic_fwhm,
        alpha_d_fm3=alpha_d,
        ewsr_reference=ewsr_reference_value,
        ewsr_exhaustion_percent=exhaustion,
        negative_area_fraction=negative_fraction,
        diagonal=spectrum.channel.diagonal,
        strength_units=spectrum.channel.strength_units,
        ewsr_label=ewsr_label,
    )
