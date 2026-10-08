# Status: 2026-08-30

## Two-EDF SLy5/SkM* result — complete

The production scope is now exactly SLy5 and SkM*. The earlier SV-bas
calculation remains below as a completed legacy stage, but it is excluded from
the new comparative figures, tables, and conclusion. No SkP-delta or broader
force survey is planned.

Sky3D's unmodified `Code/forces.data` contains the exact identifiers `Sly5`
and `SkMs`. Static HF calculations on the common 24 x 24 x 24, 1-fm grid have
converged for both EDFs:

| EDF | E_HF (MeV) | Q20 (fm^2) | beta2 | rms (fm) | final h^2 floor (MeV) | production axis |
|---|---:|---:|---:|---:|---:|---|
| SLy5 | -179.8536 | 36.532 | 0.319300 | 3.0679 | 9.427e-5 | z |
| SkM* | -180.3716 | 35.568 | 0.310877 | 3.0617 | 7.645e-5 | z |

Both raw static solutions selected the long x axis. Each production state was
rotated by the exact cubic-grid/spinor operation `R_y(-pi/2): x -> z`, without
interpolation; the maximum relative discrete norm error was 4.44e-16.

Reliable literature values used for response interpretation are
`K_infinity=229.9 MeV`, `m*/m=0.697` for SLy5 and
`K_infinity=216.6 MeV`, `m*/m=0.789` for SkM*. Sources and DOI links are stored
in `data/processed/edf_nuclear_matter_properties.{csv,json}`.

The 18-thread production queue completed the duration-matched 18000-fm/c
no-boost, E0, reverse E2(K=0), and half-strength E0 trajectories for both
EDFs. The reverse-response and half-strength analysis passed. The separately
gated SLy5 32^3 reference/E0 check is also complete; the SkM* 32^3 extension
was deliberately omitted by author decision because it is not needed for this
conference paper. The manuscript and publication figures now contain the
final two-EDF result.

Runtime inspection confirms `OMP_NUM_THREADS=18`: one main thread and 17
OpenMP workers are bound to logical CPUs 0-17, and `top -H` reports all 18
running during the propagation kernel. The additional 17 OpenBLAS pool threads
are dormant. The startup phrase `Running sequential version` refers to the
non-MPI backend in `Code/sequential.f90`; it does not mean that OpenMP is off.
Observed utilization is about 12 CPU-equivalents on average because the kernel
does not keep every worker at 100% through every operation.

The high-resolution analyzer now treats SLy5 and SkM* symmetrically, records
the boost amplitude in CSV/HDF5, and writes separate duration-convergence,
stability, reciprocity, and half-strength linearity tables. Its WSL
end-to-end smoke test produced 20 spectra and 60 summary rows from a matched
legacy pair. The publication plotting entry point has been replaced with the
requested experiment/two-EDF, SLy5 diagonal/cross, direct cross-EDF, and
compact combined layouts. All canvases are at most 6.35 inches wide and use
short panel labels.

## High-resolution response convergence

| EDF | case | grid | T (fm/c) | Rayleigh (MeV) | status |
|---|---|---:|---:|---:|---|
| SLy5 | no boost | 24^3 | 18000 | 0.068880 | complete |
| SLy5 | E0 | 24^3 | 18000 | 0.068880 | complete |
| SkM* | no boost | 24^3 | 18000 | 0.068880 | complete |
| SkM* | E0 | 24^3 | 18000 | 0.068880 | complete |
| SLy5 | E2(K=0), E0 half | 24^3 | 18000 | 0.068880 | complete |
| SkM* | E2(K=0), E0 half | 24^3 | 18000 | 0.068880 | complete |
| SLy5 | E0/no-boost box check | 32^3 | 8000 | 0.154980 | complete |
| SkM* | E0/no-boost box check | 32^3 | 8000 | 0.154980 | omitted by author decision |

The production recommendation is the 24^3, 1-fm grid and 18000-fm/c duration,
with matched no-boost subtraction. Gamma_sm=0.2 MeV is retained for response
diagnostics and Gamma_sm=1 MeV for the experiment/theory overview. Integrated
observables are the basis of the box-robust conclusion; the narrow
Gamma_sm=0.2-MeV line pattern is not claimed to be pointwise box converged.

The matched SLy5 32^3 calculation changes the 9--18-MeV E0 centroid by
0.0146 MeV and its integrated strength by 0.155%. The 9--25-MeV signed
cross-response integral changes by 0.007%. Individual narrow peaks can move by
one 0.155-MeV Rayleigh bin. The 32^3 outermost-plane density remains below
4.52e-10 fm^-3, the outer-shell population below 1.30e-7 particles, and the
particle-number drift is about 1.09e-5.

The completed E0 gate is **GO**. At Gamma_sm=0.2 MeV, the E0 and forward
cross-response characteristic energies coincide at 16.857 MeV for SLy5 and
16.59-16.60 MeV for SkM*. The signed cross-channel m0 values in 9-18 MeV are
167.67 and 174.80 fm^4, respectively. The cross-response 12000-to-18000-fm/c
relative L2 changes are 0.229% and 0.187%; maximum energy drifts stay below
0.001 MeV, and the maximum outermost-plane densities remain below
5e-8 fm^-3. Stage 2 has therefore been released for both reverse responses
and both half-amplitude checks.

Stage 2 is now **complete**. All four effective trajectories end at 18000
fm/c with `/usr/bin/time` exit status 0. The recovered SLy5 E2 trajectory used
the complete 7000-fm/c checkpoint and a strictly monotonic reconstructed
protocol; the original interrupted directories remain preserved.

At the primary `Gamma_sm=0.2 MeV` and full 18000-fm/c duration, reciprocity in
9-25 MeV is extremely tight. The forward/reverse signed-strength relative L2
difference is 0.1058% for SLy5 and 0.0957% for SkM*, with correlations
0.99999940 and 0.99999950 and best-fit reverse/forward scales 1.0001605 and
0.9999636. The complex-response relative L2 differences are 0.1075% and
0.0970%, respectively.

The half-amplitude audit also confirms linear response. In the spectral
9-25-MeV comparison, signed-strength relative L2 differences are 0.00854%
(SLy5 E0), 0.00858% (SLy5 induced Q20), 0.00906% (SkM* E0), and 0.00863%
(SkM* induced Q20). The full time-series relative L2 values are at most
0.2965%. Machine-readable results are in `reciprocity.csv` and
`linearity.csv`.

The reciprocity and linearity audit released the SLy5 32^3 E0/no-boost check,
which then completed cleanly. A SkM* 32^3 static directory had already been
created by the original queue ordering, but no matched SkM* 32^3 no-boost or
E0 production trajectory was started.

## Final paper and reproducibility products

The final manuscript is `paper/main.tex`, and the installed artifact is
`paper/output/pdf/mg24_mqc_snp2026.pdf`. It is exactly two A4 pages. Two
PdfLaTeX passes were launched from WSL; the final log contains no overfull
boxes, undefined citations, or undefined references. All fonts are embedded,
and the two publication figures are vector PDFs. Both pages were rendered at
180 dpi and inspected for clipping, legibility, and margin violations.

Figure 1 retains the absolute experimental/two-EDF comparison with no fitted
scale or energy shift. Figure 2 uses a compact four-panel response comparison;
the legends in panels (c) and (d) are stacked vertically in the empty
upper-right region and do not cover the spectra. Figure canvases are at most
6.35 inches wide, and no figure or caption exceeds the page width. The
bibliography contains five references, including the 2024 Sky3D/TDHF paper.

The refreshed high-resolution products contain 330 spectra and 990 summary
rows in CSV/HDF5 form, with `reciprocity.csv`, `linearity.csv`, and
`box_size.csv` alongside them. The WSL response-package suite passes all 10
unit/end-to-end tests, including the CSV/HDF5 schema and plot-generation test.

## Legacy SV-bas stage (superseded; not part of the two-EDF result)

The earlier milestones 1-8 are complete for the former SV-bas study. The
repository audit, deformed static reference,
no-boost controls, both independent E0/E2(K=0) response evolutions, signed
frequency-domain analysis, reciprocity, half-amplitude linearity, EWSR
normalization, smoothing robustness, and source-faithful experimental
comparison are complete. The calculation supports
the central hypothesis: the lower-energy E0 structure in strongly prolate
24Mg is accompanied by a large intrinsic K=0 quadrupole response, and the two
independent off-diagonal susceptibilities agree without rescaling. The concise
two-page SNP2026 proceedings manuscript is compiled and visually verified.

## Milestone 1: repository and convention audit

- The checkout is `be42efc` on `main`, identified by `README.md` as Sky3D v1.2
  with the 2024 multipole-response extension.
- Static/dynamic steering uses `for005` namelists. `Code/dynamic.f90` creates
  the response protocols; `Code/moment.f90` writes isoscalar and isovector
  columns to the multipole `.res` files.
- `Code/external.f90` applies `exp(-i eta F)`. For `L >= 2`,
  `F_LM=sqrt(2L+1) r^L Y_LM`; the monopole special case is
  `F00=r^2/sqrt(4 pi)`. These code conventions, rather than assumed textbook
  normalizations, are used throughout.
- A boost with one multipolarity can already be read out through every written
  multipole file. No Sky3D core change was needed for either off-diagonal
  response.
- The original `Utils/Strength_Calculation/Fourier.py` lacked reproducible
  configuration, integrated observables, and a correct continuum-time FFT
  normalization. It is now a compatibility entry point for the
  `sky3d_response` package.

The package produces signed complex diagonal/cross responses, declared
`dB(Elambda)/dE` distributions, E1 photoabsorption and dipole polarizability,
`m_-1`, `m_0`, `m_1`, `m_1/m_0`, EWSR exhaustion, and widths with the imposed
smoothing kept separate. Outputs are CSV, HDF5, JSON, vector PDF, and PNG.

Seven WSL unit/end-to-end tests pass. They cover analytic normalization,
signed cross response, reference subtraction, TRK EWSR, Fortran-number
parsing, CSV/HDF5 schemas, and plot generation. The analytic two-mode E1 test
recovers 12.005- and 18.007-MeV peaks; a 1-MeV exponential window gives 1.004-
and 1.012-MeV observed FWHM values.

## Milestone 2: static 24Mg

The isolated project executable was compiled and run in WSL with GNU Fortran
13.3.0, OpenMP, FFTW, LAPACK, and OpenBLAS. Its SHA-256 and full build command
are recorded in `logs/build_provenance.log`; no repository-wide build products
were overwritten.

The SV-bas, no-pairing calculation uses a nonperiodic 24 x 24 x 24 grid with
1-fm spacing and an unconstrained prolate oscillator seed. The HF minimum has:

- total energy: -183.8257 MeV;
- total/neutron/proton rms radii: 3.0357/3.0196/3.0518 fm;
- principal-axis Q20: 35.693 fm^2 in Sky3D's
  `sqrt(5/(16 pi)) Q_zz` convention;
- beta2 = 0.311969 and gamma = 0 degrees;
- principal mean-square axes: 4.6437, 2.2860, and 2.2860 fm^2.

Two substantially different oscillator seeds reach the same reported energy,
radius, and deformation. The raw oscillator ordering selects x as the long
axis. `scripts/rotate_wf_axis.py` applies the exact cubic-grid rotation
`R_y(-pi/2): x -> z` and the corresponding spin-1/2 rotation, without
interpolation. All discrete orbital norms are preserved to 4.45e-16 and the
production symmetry axis is z, matching Sky3D's M=0 operator.

The static iteration reaches stable bulk observables but not the requested
single-particle fluctuation threshold: `h**2` plateaus at 4.371e-5 MeV versus
`serr=1e-6`, including after continuation to iteration 5000 with lower
damping. This numerical floor is retained as a limitation.

A 50-fm/c no-boost TDHF check at `dt=0.2 fm/c` gives maximum sampled drifts of
3.8e-6 MeV in total energy, no neutron/proton-number change at 0.001 output
precision, 1.70e-6 relative in Q00, 7.69e-6 relative in Q20, and a center-of-
mass norm below 4.76e-14 fm. `figures/stationarity_svbas.pdf` and `.png` record
the control.

## Milestones 3-4: E0 boost and signed response

The baseline evolution applies Sky3D's isoscalar `F00` boost with
`eta=5e-5`. It uses `dt=0.2 fm/c`, 10000 steps, a total time of 2000 fm/c,
and multipole sampling every 2 fm/c. A duration-matched no-boost trajectory is
subtracted channel by channel before division by eta; raw, reference, and
corrected traces are all retained in CSV/HDF5 provenance.

The corrected maximum changes are 0.0062660 fm^2 for Q00 and 0.0188405 fm^2
for Q20 in the code operator conventions. The induced Q20 signal is 8.49
times the maximum duration-matched no-boost Q20 drift. It is bounded and
oscillatory. `figures/e0_time_traces.pdf` and `.png` show the traces.

The 2000-fm/c record has a Rayleigh spacing
`2 pi hbar/T = 0.619921 MeV`. Zero padding gives a 0.0774-MeV display grid but
does not improve that physical Fourier resolution. With the primary
`exp[-Gamma_sm t/(2 hbar)]`, `Gamma_sm=1 MeV` window, the 9-18 MeV analysis
interval gives:

- E0 peak and signed Q20<-E0 peak: 16.1793 MeV;
- E0 centroid `m1/m0`: 15.2397 MeV;
- E0 `m0`: 31.0181 fm^4 and `m1`: 472.7069 MeV fm^4;
- E0 EWSR exhaustion: 32.3803%;
- signed cross `m0`: +157.1756 fm^4;
- observed E0 FWHM: 1.5733 MeV;
- imposed Lorentzian FWHM: 1.0000 MeV;
- model-dependent Lorentzian intrinsic estimate: 0.5733 MeV.

In 18-25 MeV, the E0 peak is 23.8431 MeV, its centroid is 21.8586 MeV, and
its EWSR exhaustion is 25.6649%. The signed cross integral is -10.8186 fm^4
with a 0.755 negative-area fraction. The sign change is physical response
information and is never replaced by an absolute value.

Milestone-3 decision: **GO**. The low-energy E0 structure carries a clearly
resolved intrinsic-axis Q20 motion. The theoretical off-diagonal response is a
diagnostic of mixing; it is not an experimentally measured observable.

## Milestone 5: reverse E2(K=0) boost and reciprocity

The reverse run uses the same state and numerical parameters, applying the
existing isoscalar `F20=sqrt(5) r^2 Y20` boost and recording both Q20 and Q00.
For the identical kick and readout conventions, time-reversal-invariant linear
response predicts `chi_20,00(E)=chi_00,20(E)` for these two time-even Hermitian
operators.

For `Gamma_sm=1 MeV` in 9-18 MeV:

- diagonal E2(K=0) peak: 16.2567 MeV;
- E2(K=0) centroid: 15.2425 MeV;
- E2(K=0) EWSR exhaustion: 64.1532%;
- reverse signed Q00<-E2 peak: 16.1793 MeV;
- reverse signed cross `m0`: +157.0718 fm^4;
- complex-response symmetric L2 mismatch: 0.0791%;
- signed-strength symmetric L2 mismatch: 0.0713%;
- signed-strength correlation: 0.999999782;
- best real reverse/forward scale: 0.9999826.

Over 9-25 MeV the complex mismatch is 0.0777%. The matched-reference-corrected
time-domain mismatch is 1.38%. The plotted complex response curves are not
rescaled. `figures/cross_response_reciprocity.pdf` and `.png` provide the
independent cross-check.

## Milestone 6: linearity, EWSR, and robustness

The half-amplitude E0 evolution uses `eta=2.5e-5` with the same 2000-fm/c
record and matched no-boost correction. For baseline versus half amplitude:

- corrected time-domain Q00/eta mismatch: 0.2394%;
- corrected time-domain Q20/eta mismatch: 0.0118%;
- 9-18 MeV complex E0-response mismatch at 1-MeV smoothing: 0.00968%;
- 9-18 MeV complex cross-response mismatch: 0.00906%.

The maximum nonlinear residuals `2 DeltaQ_half - DeltaQ_baseline` are
3.81e-6 fm^2 for Q00 and 2.46e-6 fm^2 for Q20. The response is therefore
comfortably in the tested linear regime. `figures/e0_linearity.pdf` and `.png`
show the unscaled comparison.

The exact isoscalar double-commutator references for Sky3D's operators were
derived from the converged neutron/proton moments and SV-bas species kinetic
coefficients:

- E0 reference `m1`: 1459.8593156 MeV fm^4;
- E2(K=0) reference `m1`: 22916.8277180 MeV fm^4.

Across 1-MeV exponential, 2-MeV exponential, and cosine-power-6 windows, the
E0 MQC peak remains 16.1018-16.1793 MeV, its centroid remains
15.0410-15.4307 MeV, and the main E0 peak remains 23.7657-23.8431 MeV. The MQC
E0 EWSR interval varies from 29.54% to 35.53%; the 9-25 MeV total varies from
55.77% to 60.37%. The central identification is insensitive to this reasonable
window variation.

`data/processed/robustness_summary.csv` is the machine-readable quantitative
table. It contains beta2, EDF, boost, time step, total propagation time,
Rayleigh resolution, smoothing, peaks, moments, EWSRs, widths, reciprocity,
and linearity for every window. The cosine window has no assigned artificial
Lorentzian FWHM. All inferred intrinsic widths are explicitly labeled as
window-model dependent.

## Milestone 7: experimental comparison and final figures

The APS article records, arXiv source bundles, likely repository records, and
the Bahini thesis were searched for author-supplied numerical 24Mg IS0 data.
No complete numerical spectrum or supplemental data table was found. The 2022
authors' arXiv source does contain the vector EPS used for its experimental
spectrum. `scripts/digitize_bahini_2022_eps.py` therefore performs a separate,
fully reproducible vector-coordinate extraction from the preserved source
asset, rather than raster image tracing. The SHA-256, axis calibration,
31 extracted 0.5-MeV bins, experimental error bars, and a separate coordinate
uncertainty are recorded under `data/experimental/`.

As an independent validation, the extracted 13.75-MeV bin integrates to
37.2400 fm^4, compared with the authors' tabulated 37.7(3.8) fm^4 for the
prominent 13.87-MeV state. The agreement is not used to tune the extraction.
The experimental instrumental resolution is about 70 keV and the displayed
data are rebinned to 0.5 MeV; these are kept distinct from the TDHF Rayleigh
spacing and 1-MeV artificial smoothing.

For the absolute comparison, Sky3D's `F00=r^2/sqrt(4 pi)` strength is
multiplied by `4 pi` to match the experimental `O_IS0=sum_i r_i^2`
convention. No fitted normalization or energy shift is used. In the common
10-18 and 18-25 MeV intervals, respectively:

- experimental centroids are 15.1100 and 21.3052 MeV;
- TDHF centroids are 15.4415 and 21.8402 MeV;
- experimental integrated strengths are 158.337 and 188.765 fm^4;
- TDHF integrated strengths are 374.963 and 211.851 fm^4.

TDHF therefore reproduces the separation of a low-energy component from the
higher-energy monopole structure and gives similar broad-region centroids. It
does not reproduce the experimental fragmentation: its dominant 16.179-MeV
MQC peak is 2.43 MeV above the 13.75-MeV maximum experimental bin, and the
calculated low-energy strength is too concentrated and too large. This is the
quantitative comparison boundary used in the paper.

`figures/fig1_e0_experiment_comparison.pdf` and
`figures/fig2_dynamical_mixed_response.pdf`, with PNG companions, are the
publication figures. Their numerical comparison is recorded in
`data/processed/experimental_comparison_summary.json`.

## Milestone 8: SNP2026 manuscript

The manuscript under `paper/` uses the official SNP2026 PdfLaTeX contribution
template. The downloaded template source and SHA-256 are recorded in
`paper/TEMPLATE_SOURCE.md`; the pristine class is retained as
`paper/snp-official.cls`. The working `paper/snp.cls` changes only the obsolete
REVTeX 4.0 begin-document hook needed for current LaTeX, leaving the official
layout dimensions unchanged.

The final paper is exactly two A4 pages and contains the two reproducible
publication figures. Figure titles, legends, annotations, and captions were
shortened to avoid clutter; all figure and text content remains inside the page
and axes bounds. The five retained references cover deformation mixing, both
experimental 24Mg studies, the 2024 Sky3D/TDHF extension, and SV-bas. The 2024
TDHF paper is cited directly in the calculation method.

Both PdfLaTeX passes were launched from WSL. The final log has no overfull
boxes, undefined citations, or undefined cross-references. Poppler reports two
A4 pages, and both rendered pages were inspected at 180 dpi for clipping,
legibility, and margin violations. The installed artifact is
`paper/output/pdf/mg24_mqc_snp2026.pdf`.

## Writing revision

1. The paper now follows a physics-led chain from the experimental MQC
   component to the limitation of diagonal spectra, the induced real-time
   quadrupole motion, and the reciprocal off-diagonal response as the central
   result. The method introduces the two-channel protocol before numerical
   settings, and both captions state what each figure establishes.
2. Code-normalized EWSR percentages, the observed-width decomposition,
   alternative-window shifts, raw response amplitudes, and the static total
   energy were removed from the two-page text because they do not advance its
   central argument. They remain in the processed outputs and earlier status
   sections. The deformation was rounded to `beta2=0.312` in prose.
3. The revision keeps the cross susceptibility signed and complex and labels
   it as a theoretical diagnostic distinct from measured IS0 strength. It also
   separates the missing experimental fragmentation from the finite-box and
   Fourier-resolution limit, and retains the absence of fitted scale or energy
   shift.
4. No numerical or scientific inconsistency requiring author review was found.
   All retained values agree with the existing processed outputs; no TDHF or
   response calculation was rerun.

## Unexpected issues and limitations

- WSL Python has no `venv` module and its base environment is PEP-668 managed.
  `h5py` was installed only under
  `/home/abhishek/.local/share/sky3d-response-deps` for validation.
- The duration-matched no-boost calculation has small coherent Q20 drift
  (maximum 0.0022185 fm^2), which was not negligible at half kick strength.
  Explicit matched-reference subtraction removes this common deterministic
  background and is recorded in every processed product.
- A cosine-power window has no unique Lorentzian smoothing FWHM; no artificial
  or deconvolved width is invented for it.
- The 0.6199-MeV Rayleigh spacing, imposed damping, finite grid/box, TDHF
  mean-field dynamics, and the static fluctuation floor limit fine-structure
  and width claims. The calculation does not claim the experimental 70-keV
  resolution or a full many-body spreading width.

## Exact next action

The scoped response-analysis package, 24Mg calculation, reproducibility record,
experimental comparison, figures, and two-page proceedings paper are complete.
Any next calculation should be a separately declared extension, such as a
spherical control or a second deformed nucleus, rather than a change to this
locked result.

## Final convention and manuscript audit (2026-08-30)

No new Sky3D calculation was run. The completed 24-cubed SLy5/SkM* responses
and the completed SLy5 24-cubed versus 32-cubed check are the locked numerical
basis of the final conference paper.

The manuscript now identifies the quoted deformations as Sky3D total-mass
values: `Q20 = 36.532 fm^2`, `beta2 = 0.319` for SLy5 and
`Q20 = 35.568 fm^2`, `beta2 = 0.311` for SkM*, using
`beta2 = 4*pi*Q20/(5*A*R^2)` and `R = 1.2*A^(1/3) fm`. It also states the
actual transform convention and plotted mixed response,
`R_BA = Im chi_BA^(-)/pi`, with units `fm^4 MeV^-1`. Off-diagonal responses
are explicitly signed and are not treated as positive-definite strengths.

At 0.2-MeV artificial smoothing, the saved complex reciprocity mismatches are
0.108% for SLy5 and 0.097% for SkM*. The new text connects this check to the
Onsager--Kubo condition for time-even Hermitian operators. The calculated
centroid shifts relative to experiment are `(0.627, 0.304) MeV` for SLy5 and
`(0.416, 0.452) MeV` for SkM* in the 10--18 and 18--25 MeV windows.

The box result is stated as a convergence hierarchy: the 9--18 MeV E0
centroid and integral shift by 0.015 MeV and 0.15%, and the signed mixed
integral shifts by 0.01%, while the narrowly smoothed pointwise E0 spectrum
has a 79% relative L2 difference and individual peaks can move by one
0.155-MeV Rayleigh bin. The coupling claim therefore rests on reciprocity,
linearity, EDF persistence, and stable window observables, not on a signed
mixed integral, fine line positions, or intrinsic widths.

The final figures were regenerated from existing CSV data with compact labels
and vertically stacked upper-right legends in Figure 2(c,d). The final paper
uses five numbered references, retains the 2024 Sky3D/TDHF paper, and adds the
Kubo reciprocity source while replacing the generic EDF citation by the
original SLy5 and SkM* papers.

Two PdfLaTeX passes were launched from WSL. The installed artifact is exactly
two A4 pages, has no overfull boxes or undefined references, uses embedded
non-Type-3 fonts and vector figures, and passed rendered-page inspection at
180 dpi. The final PDF is `paper/output/pdf/mg24_mqc_snp2026.pdf`, with SHA-256
`709e47ceda338c404c930ba3807ef8f01aea60da2b924354682db5ea4ff435bb`.

## Two-EDF extension: outage-recovery record (2026-08-30)

A power outage interrupted `e2k0_sly5_t18000` after formatted protocols reached
7454 fm/c; its last complete restart wave function was iteration 35000 at
6999.9999999958 fm/c. The original interrupted directory remains unchanged.
The first staging diagnostic, `e2k0_sly5_t18000_recovery1`, was intentionally
stopped at 7046 fm/c after exposing the legacy `dipoles.res` header that labels
physical time as `Iter`; it remains diagnostic evidence.

The production recovery, `e2k0_sly5_t18000_recovery2`, resumed from the complete
7000-fm/c checkpoint with 18 OpenMP threads and excluded duplicated checkpoint
rows by inferred protocol scale. The recovered SLy5 E2 trajectory, SkM* E2
trajectory, and both half-amplitude E0 trajectories subsequently reached
18000 fm/c. Their completed analysis produced `reciprocity.csv`,
`linearity.csv`, `spectra.csv`, `summary.csv`, and
`response_convergence.h5`; these are the stage-2 products used in the final
audit above.
