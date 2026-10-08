# 20Ne K-resolved isoscalar E2 response

## ABSTRACT READY: YES — with a finite-time-resolution qualification

The 20Ne SLy5 calculation gives a clear ordering of the giant-quadrupole
strength across the three intrinsic components: the K=0, K=1, and K=2
dominant structures occur at approximately 16.55, 19.93, and 21.66 MeV in the
10–35 MeV interval. The half-strength K=0 check agrees with the baseline at
the 0.054% relative-L2 level, supporting linear response. The result is safe
for an ICNPA abstract as a calculated deformation-induced redistribution and
K-dependent ordering, but the 6000 fm/c run has a 0.207 MeV Rayleigh spacing
and should not be presented as an experimentally resolved width measurement.

## Numerical setup

- Sky3D v1.2, SLy5, no pairing, 20Ne (`N=Z=10`).
- Cartesian grid: 24 × 24 × 24 points, spacing 1 fm, non-periodic Coulomb.
- Static solver: existing DAE protocol (`x0dmp=0.40`, `e0dmp=100`, `serr=10^-6`,
  maximum 3000 iterations).
- TDHF: `dt=0.2 fm/c`, 30,000 steps, endpoint 6000 fm/c, fourth-order Taylor
  propagation, `eta=5×10^-5`; K=0 half test uses `eta=2.5×10^-5`.
- All five trajectories were stopped by the WSL endpoint watcher at exactly
  6000 fm/c. Analysis uses exact protocol prefixes ending at that time.
- Exponential analysis smoothing: Γ_sm=0.5 MeV. This is imposed smoothing,
  not a physical spreading width.
- Rayleigh spacing: 0.2066 MeV.

## Static ground state

The converged static state has total energy `-141.9926 MeV`, total rms radius
`2.9692 fm`, principal `Q20=28.146 fm²`, and `β≈0.3334`, `γ=0°`. It is
prolate. The converged seed was elongated along x; the existing exact cubic
grid/spinor rotation `R_y(-π/2)` maps that axis to z without interpolation.
The maximum discrete norm error of the 20 rotated spinors was
`3.3×10^-16`.

## E2 operator definition

Sky3D applies `ψ(0+)=exp[-i η F_LM]ψ(0)` with equal neutron and proton factors
for `isoext=0`. For `L≥2`, the implemented operator is

`F_LM = sqrt(2L+1) r^L Y_LM`.

For positive M, this checkout's `Y_lm` routine returns the real cosine
component and does not include the additional √2 used in a unit-normalized real
tesseral convention. The raw code-native K=1 and K=2 spectra are therefore
converted by a factor of two in diagonal strength for the comparison table;
the conversion is recorded in `summary/results.json` and `results.csv`.

## K-resolved results

| K | Dominant peak(s), 10–35 MeV | Centroid (MeV) | m0 (fm⁴) | Notes |
|---:|---|---:|---:|---|
| 0 | 16.55, 17.64, 19.00 | 17.85 | 911.2 | strongest high-energy branch; additional fragmentation |
| 1 | 19.93, 18.77, 19.31 | 19.55 | 719.7 | intermediate branch, fragmented around 19 MeV |
| 2 | 21.66, 22.23, 23.34 | 22.54 | 467.6 | highest-energy branch and weakest integrated strength |

The ordering `K=0 < K=1 < K=2` is the strongest robust numerical result in
this finite-time calculation. A lower-energy structure near 6 MeV is visible
in the raw spectra for K=0 and K=1; it is excluded from the giant-quadrupole
table rather than being folded into the branch ordering.

## Linearity check

The normalized K=0 baseline and `eta/2` responses were compared over
0.5–35 MeV. Relative L2 difference is `5.40×10^-4`; maximum absolute
difference divided by the baseline peak is `5.03×10^-4`. This supports use of
the weak-field response normalization.

## What is demonstrated

Within SLy5 TDHF, a prolate, z-aligned 20Ne state redistributes isoscalar E2
strength among the code-resolved K components and produces a clean upward
ordering of the dominant giant-quadrupole structures from K=0 through K=2.
The calculation is internally linear at the tested boost amplitude.

## What should not be claimed

This is not an experimental comparison, an EDF-independent prediction, a
continuum-corrected width, or a proof that every fine peak is physically
resolved. The K=1 and K=2 comparison uses an explicitly documented tesseral
normalization conversion. The finite 6000 fm/c interval limits the energy
resolution to 0.207 MeV, and Γ_sm=0.5 MeV is artificial.

## ICNPA recommendation

20Ne alone is sufficient for a compact abstract centered on deformation-induced
K splitting in TDHF. Quote the peak ordering and centroids above, together with
the finite-time/smoothing qualification. A SkM* repeat would test EDF
robustness; an oblate 28Si comparison would provide stronger shape-systematics,
but neither is required before presenting this first result.

