# 20Ne static convergence study, 2026-10-08

The supplied 24³, 1 fm Sly5 calculation has a mesh-dependent convergence
floor. A 40³, 0.6 fm calculation in the same 24 fm box, followed by extra
settling, meets the original `serr=1e-6` threshold in both the saved
pre-gradient measure and an independent fresh-field check. This is a better
candidate for further CPU/GPU comparisons; it is not a validated response
spectrum or a new speedup measurement.

The production input, Skyrme force, Hamiltonian, derivative rules, propagator
and existing production outputs are unchanged. Static calculations remain CPU
calculations. The only production source change in this study reports whether
static iteration met its criterion or exhausted its limit.

## Measured residuals

The diagnostic rebuilds all densities and mean fields from a checkpoint,
then applies the actual Fortran Hamiltonian before any gradient update.
It measures both the historical `sqrt(abs(<(H-e)^2>))` quantity and the
direct residual norm `||(H-e)psi||`, with `e=Re<psi|H|psi>/<psi|psi>`.
The quantities below are occupation-weighted means per propagated state
(20 fully occupied states), in MeV. These are different measures when the
discretized Hamiltonian is not exactly Hermitian.

| Grid, spacing | Saved iteration | Fresh H² fluctuation | Fresh direct residual | Integrated energy, MeV |
|---|---:|---:|---:|---:|
| 24³, 1 fm | 3000 | 7.559e-5 | 1.223e-4 | -141.995969 |
| 32³, 0.75 fm | 3000 | 2.629e-6 | 2.157e-6 | -141.979175 |
| 40³, 0.6 fm, settled | 1200 | 1.101e-7 | 1.598e-8 | -141.980365 |

All grids have `n*dx=24 fm` in each direction, with the same interaction,
particle numbers, initial oscillator radii and damping parameters. The 24³
checkpoint is from the earlier optimized CPU run; all fresh-field probes and
the 32³/40³ iterations use strict CPU arithmetic. A change in orbital phase
or occupied-space basis does not establish a different physical state.

The 40³ state's largest individual direct residual is `5.183e-8 MeV`;
its largest individual H² fluctuation is `1.928e-7 MeV`. Rebuilt particle
numbers are 10 neutrons and 10 protons within `8e-13`, and the largest
occupied-basis orthonormality error is below `3e-13`. The integrated energy
changes by about 1.19 keV between 32³ and 40³. Additional grid/box tests and
time-dependent physical observables are still needed for production claims.

## Source of the plateau

For each isospin, the probe constructs the entire occupied-space matrix
`Hij=<psi_i|H|psi_j>`, including both triangles. On 24³, the largest entry of
`H-H†` is `4.373e-4 MeV` for neutrons and `3.987e-4 MeV` for protons.
Independently reconstructing the effective-mass term with NumPy FFTs
accounts for this difference; after subtracting that contribution, the
remaining terms' anti-Hermitian parts are below `2e-14 MeV`.

The source applies the effective-mass term in expanded form,
`-(dB/dx)(dpsi/dx)-B(d²psi/dx²)`, summed over directions. Its discrete
derivatives do not obey an exact continuum product rule. The independently
reconstructed field derivatives agree with the source's derivative matrices
to about `2e-13`. Thus a mismatch between the FFT and matrix derivative
implementations does not explain the observed effect in this case.

Mesh refinement reduces the occupied-space anti-Hermitian entry to about
`5.1e-6 MeV` on 32³ and `4.55e-8 MeV` on 40³. Together with the residual
measurements, this supports a discretization explanation for this state's
plateau. These measurements are projected onto the occupied space; they
are not a bound on the full Hamiltonian's operator norm. The calculation
does not alter the effective-mass operator to force Hermiticity.

## First stop and settling

The 40³ run first met the program's original pre-gradient criterion at
iteration 878 (`9.986e-7 MeV`). Rebuilding fresh fields gave an H² quantity
of `1.459e-6 MeV`, although the direct residual was only `3.024e-7 MeV`.
Meeting that first stopping test alone did not meet the fresh H² test.

A restart with the original criterion stopped at 881. A separate copy was
then continued to 1200 using an explicitly stricter study target of `1e-8`.
That stricter target was not met (`8.863e-8` in the saved measure), but the
original `1e-6` target is met in both saved and fresh fields, as shown above.
The study does not silently relax a tolerance or certify the stricter target.

The first-stop cache checkpoint was overwritten through a restart symlink;
its original logs and raw probe vectors remain. The first-stop JSON explicitly
records that retention limit. The settled checkpoint is a separate regular
file. The reusable script always copies checkpoints before continuing them
and checks that the supplied file's hash remains unchanged.

## Reproduce in WSL/Linux

Only gfortran, make, FFTW3, LAPACK/OpenBLAS, Python and NumPy are needed for
this static diagnostic; CUDA is unnecessary. The script builds unmodified
strict CPU and instrumented probe executables under a fresh `~/.cache`
directory. Instrumentation stops immediately after fresh field construction;
it never changes `Code/` or evolves the supplied checkpoint unless the user
explicitly requests settling of a copy.

```bash
cd /mnt/d/Coding/sky3d
python3 CUDA/static_convergence.py --mesh 32 \
  --output /tmp/static32.json
python3 CUDA/static_convergence.py --mesh 40 --maxiter 2500 \
  --settle-iterations 400 --settle-serr 1e-8 \
  --output /tmp/static40.json
python3 CUDA/static_convergence.py --state /path/to/20ne_sly5.tdhf \
  --output /tmp/static-probe.json
```

The 400 additional iterations above are a reproducible settling recipe,
not an assertion that every machine will stop at exactly the measured
iteration. The saved inputs, compiler flags, source hashes, executable
hashes, checkpoint hashes, residuals and local directories are recorded in
`CUDA/results/static-convergence-*.json`. Raw wavefunctions, probes and
logs remain outside Git. The local settled checkpoint is:

```text
/home/abhishek/.cache/sky3d-convergence-x6wxa6r1/polished40/20ne_sly5.tdhf
```

The actual CUDA workspace query estimates **0.367 GiB** for 40³/20 states,
comfortably within this GPU's queried 10.75 GiB free memory. That query does
not demonstrate GPU correctness or speed on the finer grid. The existing
`benchmark.py` remains specific to the 24³ input and should not be given
this checkpoint until its grid/spacing handling and validation are extended.

## Static exit diagnostics

`statichf` now prints an explicit success message or an iteration-limit
warning, plus the final pre-gradient H²/H†H quantities and requested `serr`.
The legacy exit code and stopping definition are preserved. Callers still
need to check convergence; successful process exit alone is insufficient.
`check_static_status.py` exercises success and exhaustion with `mprint=0`
and `mprint=10`, checks the saved iteration and reported fluctuation, and
compares wavefunctions against the pre-change strict CPU executable.
Maximum differences are below `1e-12`; numerical evolution is unchanged.
An additional three-step settling check verifies input-checkpoint preservation.
See `CUDA/results/static-status.json`.
