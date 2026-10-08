# Workflow notes

## Validated reference workflow

The reference calculation is `projects/mg24_mqc_snp2026`. The files inspected
before preparing this project were:

- `configs/static_24mg_sly5.in`: SLy5, no pairing, 24^3 Cartesian grid at
  1 fm spacing, damped-gradient static solver with `tdiag=T`, `x0dmp=0.40`,
  `e0dmp=100`, `maxiter=3000`, and `serr=1e-6`;
- `scripts/rotate_wf_axis.py`: exact cubic-grid `R_y(-pi/2)` index/spinor
  rotation used when a converged long x axis must be mapped to z;
- `configs/e2k0_highres_24mg_sly5.in`: instantaneous isoscalar boost,
  `eta=5e-5`, `dt=0.2 fm/c`, fourth-order Taylor propagator, and protocol
  output every 2 fm/c;
- `scripts/run_response_wsl.sh`, `run_response_restart_wsl.sh`, and
  `prepare_response_restart.py`: non-overwriting WSL runs, timed logs, and
  checkpoint-safe continuation;
- `Utils/Strength_Calculation/sky3d_response`: baseline subtraction, explicit
  time-integration factor, signed response, exponential smoothing, moments,
  CSV/HDF5 metadata, and publication plots;
- `scripts/analyze_highres_convergence.py` and publication plotting scripts:
  duration/smoothing checks, relative-L2 linearity checks, compact figure
  dimensions, and separate numerical versus imposed widths.

## Sky3D operator and response convention

`Code/external.f90` applies the instantaneous boost

`psi(0+) = exp[-i eta F_LM] psi(0)`

with equal neutron and proton factors for `isoext=0`. For `L>=2`,

`F_LM = sqrt(2L+1) r^L Y_LM`.

`Code/ylm.f90` returns `Re Y_lm` for positive m and an imaginary/sine
component for negative m. It does not multiply the nonzero-m components by
`sqrt(2)`. Thus M=1 and M=2 excite cosine-type real tesseral components with
code-native normalization. In an axial z-aligned state the sine partner is
degenerate by rotation about z. Raw spectra remain in the code convention;
any conversion to unit-normalized real tesseral operators is labeled and uses
a factor of two in diagonal strength for `|M|>0`.

`Code/moment.f90` evaluates the readout with the same
`sqrt(2L+1) r^L Y_LM` convention about the instantaneous center of mass and
writes time, isoscalar value, and isovector value to `quadrupoles.res`.
The transform convention is

`I(E)=integral dt exp(-iEt/hbar) delta<Q>(t)`,
`R(E)=I(E)/(eta hbar)`, and `S(E)=Im R(E)/pi`

for Sky3D's negative boost phase. Exponential smoothing is
`exp[-Gamma_sm t/(2 hbar)]`, so `Gamma_sm` is the artificial Lorentzian FWHM.
The Rayleigh spacing is `2 pi hbar/T`; zero padding only interpolates.

## Alignment and run choice

The 20Ne static seed is elongated along z. Its converged Cartesian second
moments and principal-axis eigenvectors must be checked before TDHF. If the
long axis is not z, the already validated exact cubic-grid rotation script is
reused; no interpolating rotation or Fortran physics change is permitted.

The production duration was shortened at the user's request to 6000 fm/c
(30000 steps), giving a Rayleigh spacing of about 0.207 MeV. All K trajectories use
identical numerical settings and start from the same static wavefunction.
