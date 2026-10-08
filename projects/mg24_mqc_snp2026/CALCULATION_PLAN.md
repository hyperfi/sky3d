# Two-EDF calculation plan

## Scope and decision rule

Use only the standard Sky3D SLy5 (`Sly5`) and SkM* (`SkMs`) implementations.
The goal is robustness across these two EDFs, not general EDF independence and
not a broad force survey. SV-bas output is legacy-only; SkP-delta is out of
scope.

## Execution gates

1. **Implementation audit — complete.** Verify the two force-table entries,
   executable provenance, and unchanged parameters.
2. **Static HF — complete.** Converge prolate 24Mg with identical grids and
   minimization settings; rotate the exact cubic-grid solutions from long x to
   production z without interpolation.
3. **E0 comparison — running.** For each EDF run a matched 18000-fm/c no-boost
   reference and E0 trajectory; inspect Q00, Q20, stability, duration prefixes,
   and smoothing before releasing the reverse calculation.
4. **Reverse response.** If both E0 calculations are stable and exhibit a
   usable mixed response, run E2(K=0) to 18000 fm/c for both and test
   `chi_20,00 = chi_00,20` without rescaling.
5. **Linearity.** Run eta/2 E0 trajectories to the same final time for both;
   compare corrected response per eta in time and frequency domains.
6. **Continuum audit.** Track density-integrated particle number, rms radius,
   outer-shell population, and outermost density. Run a same-spacing SLy5
   32^3 box check at 8000 fm/c (Rayleigh spacing 0.155 MeV), with a separately
   converged reference and matched no-boost subtraction. The author ended the
   box survey after SLy5; no SkM* 32^3 dynamic response is required for the
   conference paper.
7. **Final analysis.** Use T = 2000, 4000, 8000, 12000, and 18000 fm/c prefixes
   and Gamma_sm = 1.0, 0.5, 0.2, 0.1, and, if window damping is adequate,
   0.07 MeV. Keep Rayleigh spacing, zero-padded bin spacing, smoothing, and
   physical structure separate.
8. **Outputs.** Produce reproducible CSV/HDF5, compact tables, and page-safe
   vector figures. Rewrite the manuscript only after all numerical gates pass.

The stage-2 launcher requires a human-readable `GO` gate written only after
the completed SLy5 and SkM* E0 spectra, matched references, and stability
metrics have been inspected. No automatic queue can bypass this comparison.

## Required comparison products

- static energy, intrinsic Q20, beta2, total/neutron/proton rms radii,
  orientation, and convergence floor;
- low-energy E0 centroid/characteristic energy, associated K=0 E2 energy,
  signed cross-response energy, and main ISGMR centroid;
- integrated strengths and EWSR exhaustion only in explicitly declared energy
  windows;
- reciprocal-response mismatch and half-strength linearity mismatch;
- literature K_infinity and isoscalar m*/m;
- propagation, smoothing, no-boost, density-edge, and box-size stability.

Fragmented spectra are summarized with fixed-window centroids rather than a
forced single peak label.
