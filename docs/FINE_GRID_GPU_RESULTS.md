# Finer-grid 20Ne GPU checkpoint

This checkpoint validates the CUDA C++ backend against the settled 40³,
0.6 fm Sly5 20Ne state. The state and grid are shared between CPU and GPU
runs. Static preparation remains on the CPU. Raw checkpoints, density files,
logs and instrumented builds remain in WSL's cache; Git contains only source,
documentation and small JSON results.

## Repeated complete-job timing

On the local RTX 5070 / Core Ultra 7 265K, the GPU median is **43.45 seconds**
against **62.60 seconds** for the best measured CPU configuration: **1.441×
speedup**, or **30.6% less wall time**. Each job runs 200 steps at `dt=0.1`
(20 fm/c), starting from the identical checkpoint. This ratio compares the
same finer-grid workload on CPU and GPU; it does not compare with the older
24³, `dt=0.2` calculation.

| Backend and CPU-side threads | Run 1, seconds | Run 2, seconds | Run 3, seconds | Median, seconds |
|---|---:|---:|---:|---:|
| CPU, 4 | 65.659 | 66.768 | 65.617 | 65.659 |
| CPU, 8 | 62.550 | 62.597 | 63.944 | 62.597 |
| CPU, 20 | 85.874 | 85.844 | 85.861 | 85.861 |
| GPU, 8 | 43.454 | 43.463 | 42.944 | 43.454 |

Jobs run sequentially after separate warmups, with rotated/reversed order.
Startup, initial/final checkpoints and diagnostics every 10 steps are
included. Density plots are disabled identically. Profiling and memory
checking are separate runs. Every timed configuration's final wavefunctions
and printed observables were compared with the best CPU configuration from
the same repetition. Maximum wavefunction difference is `2.36e-16`; maximum
particle drift from 10 per species is `3.07e-10`.

These are three local repetitions per configuration, with a short physical
endpoint. They establish a measured speedup for this workload, not a universal
optimal thread count or a production-length accuracy/performance guarantee.
Full precision timings, inputs' hashes, executable hashes and output checks
are in `CUDA/results/20ne-40x40x40/benchmark.json`.

## Integration settings and limits

The original `dt=0.2 fm/c` is unstable on this finer mesh, including in the
strict CPU reference. It stopped around 11 fm/c with a quadrupole
diagonalization failure after particle loss and energy growth. The runner
now rejects a missing requested endpoint even if a legacy Fortran `STOP`
returns process exit code zero. The failed case is preserved in
`CUDA/results/20ne-40x40x40/unstable-dt02.json`.

All successful comparisons use the explicit smaller `dt=0.1 fm/c`, retaining
Taylor order 4 and the laboratory quadrupole pulse `L=2, M=0, amplitude=5e-5`.
The finer state is elongated along x. Although the input template is named
`td_k0.in`, these runs do not establish an intrinsic K=0 response spectrum.
The state geometry is recorded separately in `state-geometry.json`.

The rectangular-grid control measured an initial CPU fast-math versus strict
CPU collective-flow difference of `7.893e-6 MeV` for neutrons and
`6.052e-6 MeV` for protons. This is the sensitive `j²/rho` diagnostic in padded
near-vacuum cells. Its explicit time-zero comparison tolerance is `1e-5 MeV`.
At later times the existing `5e-6 MeV` tolerance remains; total/integrated
energy, wavefunction, field and moment tolerances are unchanged. The default
time-zero tolerance remains `5e-6`; this case chooses its override explicitly.
The CPU-only calibration is saved in `initial-flow-roundoff.json`.

## Numerical validation

Three 100-step cases compare optimized CPU and GPU against a one-thread
strict CPU executable: 40³, padded 44×42×40 at the same spacing, and a 40³
case with center-of-mass correction every step. All passed. Across these
comparisons:

- Maximum complex wavefunction error: `2.830e-14` absolute,
  `9.552e-14` relative L2.
- Maximum density/potential field error: `3.823e-12` absolute.
- Total/integrated energies agree at the saved output's printed precision.
- The integrated two-step GPU memory checker reports zero errors.
- Shared-checkpoint restart from step 20 to 40 passes; maximum wavefunction
  error is below `2.8e-16`.

Validation jobs are distinct from benchmark jobs. Completed validation
outputs were reused when calibrating the initial flow comparison; their
inputs and final states were retained and all comparisons recomputed. Their
elapsed times are not used for speedup claims. See `validation.json`,
`sanitizer.json` and `restart.json` in the case's results directory.

These are implementation checks on short trajectories, not a long physical
response calculation, an independent experimental comparison, or proof for
every nucleus, force, pairing option and machine. The timestep evidence is
specific to this case and does not establish a universal stability bound.

## Stage profile

One separate instrumented 100-step job per backend, with 8 CPU-side threads,
locates the remaining costs. Instrumented wavefunctions and printed
observables were checked against the passed uninstrumented jobs.

| Stage | CPU seconds | GPU seconds |
|---|---:|---:|
| Predictor, including density accumulation/download | 7.14 | 3.18 |
| Midpoint fields and upload | 4.04 | 3.47 |
| Corrector, including density accumulation/download | 12.55 | 5.67 |
| Diagnostics | 5.15 | 4.73 |
| End-of-step fields and upload | 4.00 | 3.48 |

Field construction/upload and diagnostics occupy about 11.68 seconds of the
22.75-second instrumented GPU job. These single-job timings identify costs;
only the repeated uninstrumented jobs establish speedup. The timings include
CPU/GPU synchronization and transfers. Initialization, program setup and
output are included in the full job but not all are represented in this table.

The nested GPU-job Skyrme timer measures 6.76 seconds, including 1.24 seconds
in Coulomb. These timers overlap the field stages and must not be added to
them. Coulomb alone is a relatively small part of the full job; moving the
complete field calculation is a stronger GPU target than moving only Poisson.

The next useful options are:

1. Keep field construction on the GPU with the existing resident densities,
   implementing the current spectral derivatives and Skyrme algebra in CUDA
   C++. This targets about 7 seconds of the instrumented job. It needs full
   force/field equivalence tests and a revised allocation/workspace guard.
2. Parallelize the CPU diagnostic state loops or implement their reductions
   on GPU. `tinfo` and `sp_properties` currently perform substantial orbital
   work on the CPU. The whole diagnostic stage costs about 4.7 seconds here;
   that is an upper bound on what any such change could remove, not a promised
   saving. Per-component profiling and matched tests should precede a claim.

The Hamiltonian, mean-field discretization and production diagnostic loops
are unchanged by this checkpoint. Profiling instrumentation lives only in
temporary source copies. See `profile.json` for stage counts, inclusive
timings, source hashes and instrumentation limits.

## Reproduce

The runner reads the checkpoint's grid and spacing and keeps non-24³ results
separate. Regenerate a finer state using `CUDA/static_convergence.py` as
described in [the static study](STATIC_CONVERGENCE.md), then run the explicit
fine-grid commands in [the CUDA README](../CUDA/README.md).

The measured state hash is
`ea7ec8de58dcbe1d37a3c003cffa263b1a58542af27d3abc9f99078c5cea5abc`.
GPU memory queries estimate 0.367 GiB of arrays/workspace for this case; the
runtime reserves headroom and refuses oversubscription. Fitting in VRAM is
not a speedup guarantee. Rebuild, rerun preflight, and retune thread counts
when cloning the branch on another machine.
