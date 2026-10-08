# Integrated CUDA C++ pilot: local results

Measured in WSL on 2026-10-08: RTX 5070 (12227 MiB), driver 610.88,
CUDA 13.0, gfortran 13.3, Intel Core Ultra 7 265K. The branch is `gpu`.
The Fortran frontend calls an FP64 CUDA C++/cuFFT backend through a C ABI.
Propagation, Hamiltonian application, spectral derivatives and density
accumulation are integrated; fields, moments, diagnostic work and I/O remain
on CPU. The pilot is single-process and requires `tfft=T`.

## Complete-job timing

20Ne, SLy5, 24³ grid at 1 fm spacing, 20 propagated states, 500 TDHF steps
at 0.2 fm/c (100 fm/c), Taylor order 4, E2 impulse amplitude 5e-5.
Each configuration has three sequential, interleaved complete jobs after
separate warmups. Startup, initial/final checkpoints and diagnostics every
10 steps are included; density output is disabled equally for timing jobs.
Validation and sanitizer overhead are excluded. OpenBLAS uses one thread.

| Initial state | CPU configuration | CPU median | GPU median | CPU/GPU ratio |
|---|---|---:|---:|---:|
| Existing local aligned state | Best of 4/8/20 threads: 8 | 20.507 s | 18.731 s | 1.095× |
| Fresh benchmark state | 8 threads selected by screening | 20.193 s | 18.774 s | 1.076× |

The fresh state's wall time falls by about **7.0%**, and the existing state's
by about **8.7%**. These are modest measured end-to-end gains on this machine,
not the roughly 2× resident derivative-kernel gains from the historical audit.
The fresh-case repeats span CPU 20.171–21.215 s and GPU 18.735–19.174 s.
Three repetitions on a shared desktop do not establish a universal speedup
or precise confidence interval. Larger nuclei and other machines are unmeasured.

The CPU-thread screening found medians 25.074 s (4), 20.507 s (8), and
20.906 s (20) for the existing state. GPU CPU-side work used eight threads.
CUDA graph replay is enabled in the final backend. The pre-graph integrated
run measured 20.975 s on its best CPU configuration and 18.987 s on GPU;
host variation prevents assigning the small timing change solely to graphs.

Evidence: `CUDA/results/benchmark.json` (fresh state),
`benchmark-screening.json` (existing state and thread screening), and
`benchmark-launches.json` (pre-graph). Each records state/source hashes,
flags, hardware and raw-run directories. Source commits anchor builds made
with pending GPU changes; source hashes identify the measured implementation.

## Numerical checks

- Three 100-step cases against a strict, one-thread CPU reference: 24³,
  rectangular 28×26×24, and `mrescm=1`. Final complex wavefunction relative
  L2 differences are about 1e-14. All rho, tau, current, spin, spin-orbit,
  potential and Coulomb fields pass recorded tolerances.
- Initial GPU Hamiltonian comparisons are about 1.5e-15 relative L2. These
  checks are separate from timing runs.
- All three 500-step GPU checkpoints match the optimized CPU trajectories.
  Checkpoint particle-number drift is below 5e-8 for this trajectory; CPU and
  GPU share the finite-order propagator's drift.
- Total and integrated printed energies pass an absolute 2e-7 MeV comparison.
  Moments and printed kinetic energy use tolerances that account for their
  output precision. The small `j²/rho` collective-flow diagnostic is sensitive
  to near-vacuum padding: CPU fast-math versus strict CPU alone differed by
  about 1.4e-6 MeV at time zero. Only that diagnostic uses a 5e-6 MeV absolute
  tolerance; wavefunction/field/total-energy tolerances are not relaxed.
- A shared strict-CPU checkpoint resumed from step 20 to 40 on all three
  executables passes wavefunction, field and observable comparisons.
- CUDA 13's Compute Sanitizer reports zero memory errors on an integrated
  two-step GPU trajectory, including the initial primitive checks.
- Both existing CPU Makefiles compile without CUDA. Control regressions
  cover center-of-mass accumulation, disabled output, long fragment paths
  and restart intervals. Static comparisons include repeat thread-count
  checks and printing disabled; full field differences are below 4e-13.

Details and exact tolerances are in `validation.json`, `restart.json`,
`sanitizer.json`, `static.json`, `cpu-builds.json`, and
`tools/gpu_audit/evidence/controls-fixed.json`. These are implementation and
short-trajectory comparisons, not a new production response-spectrum validation.

## Static-state limit

Fresh preparation uses the existing 20Ne static input and an **explicit**
`--static-serr 1e-4` benchmark override. The resulting state meets that criterion
at iteration 834, with weighted fluctuation 9.984e-5. Its hash is shared by the
fresh-case validation and timing evidence. The production input remains unchanged.

The original `serr=1e-6` criterion was not met after 3000 iterations, either
before or after repairing the static matrix-read race; fluctuations were
about 7.56e-5. The reason for that stricter plateau remains unresolved. The
script rejects an unmet criterion and never lowers it automatically. This
benchmark does not certify a production ground state. See `static-strict.json`
and [CPU_FIXES.md](CPU_FIXES.md) for the separate CPU corrections.

## Memory guard and portability

Actual cuFFT workspace queries on this toolkit report these array requirements:

| Planned grid and propagated states | Required device arrays + cuFFT workspace | Local preflight |
|---|---:|---|
| 24³, 20 | 81.2 MiB | Fits |
| 48³, 208 | 6.211 GiB | Fits at the queried free memory |
| 64³, 208 | 14.723 GiB | Refused on the 12 GB GPU |

The 208-state rows are memory-planning queries, not nuclear calculations or
runtime measurements. Workspace can change with dimensions, toolkit and GPU.
The runtime checks currently free memory, reserves 20% by default, warns
above 60% use, and refuses oversubscription. Simulated free budgets of 8 GiB
and 6 GiB exercise pressure and refusal respectively for the 48³/208 case.
The oversized runtime-guard test exits before allocating nucleus arrays.

VRAM fit is not a runtime prediction. Paging or repeated transfers in another
implementation may be slower than a parallel CPU job; this implementation
does not page its working arrays into host memory. Compare matched complete
jobs on each target machine. Host RAM, CPU field work, GPU FP64 performance,
diagnostic frequency and input shape can all matter.

[CUDA/README.md](../CUDA/README.md) gives build, fresh-state preparation,
benchmark, preflight and sanitizer commands. Rebuild native binaries after
cloning on another machine. Large checkpoints, `.tdd`, objects, modules,
executables and raw logs are ignored; only source and small JSON evidence
are committed. The branch is local until explicitly pushed.
