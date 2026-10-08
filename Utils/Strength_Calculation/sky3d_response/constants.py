"""Physical constants used by the response analysis."""

from math import pi

HBARC_MEV_FM = 197.3269804
FINE_STRUCTURE = 7.2973525693e-3
HBAR2_OVER_2M_NUCLEON_MEV_FM2 = 20.73553

# sigma_gamma(E)[mb] = E1_PHOTOABSORPTION_MB * E[MeV]
#                      * dB(E1)/dE[e^2 fm^2 / MeV]
E1_PHOTOABSORPTION_MB = 10.0 * (16.0 * pi**3 / 9.0) * FINE_STRUCTURE
E2_MEV_FM = FINE_STRUCTURE * HBARC_MEV_FM
