# Energy diagnostic fix and 20Ne quadrupole reproduction

Work on branch `gpu` fixes a shared CPU/GPU diagnostic problem. Fresh repeated
timing gives 1.450× GPU speedup on the settled fine-grid 20Ne case. The full
6000 fm/c K=0 GPU response reproduces the current CPU and saved calculation.
Builds and all numerical analysis use WSL; raw output stays
in its cache. The existing research project is read-only during this work.

## Diagnostic correction

After each corrector step, `Code/dynamic.f90` now refreshes the mean fields
before calling `tinfo`. Integrated Coulomb and single-particle energies and
density-file potentials use the endpoint density and fields together. The
existing Skyrme call moved; no extra evaluation was added.

For time-dependent external pulses, the internal scalar potential is preserved
while diagnostics evaluate the field at the reported time. Restoring that
internal potential then adding the original next-predictor external field
preserves propagation timing without subtractive cancellation. A nonzero-time
restart reconstructs the same predictor field as uninterrupted evolution.
The instantaneous-boost response uses no extra scalar-potential buffer.

`CUDA/validate_energy_fix.py` verifies the change using the settled 40³/0.6 fm
20Ne state. Kick and Gaussian-pulse trajectories agree with the archived strict
CPU executable to maximum wavefunction differences of 1.76e-16. Current CPU
and GPU wavefunctions, densities, fields and printed observables agree with the
new strict CPU reference. Fresh endpoint Coulomb potentials agree with an
independent recomputation to 2.67e-15 MeV.

Diagnostic intervals 0, 1 and 7 leave pulse evolution unchanged. Two restart
boundaries at steps 20 and 30 reach step 40 with CPU/GPU wavefunction differences
below 2.3e-16 from uninterrupted strict CPU evolution. The small report is
`CUDA/results/20ne-40x40x40/energy-fix-validation.json`. The initial invalid
Gaussian fixture was rejected by the endpoint check and retained in cache;
the corrected fixture uses tau0=2.5 fm/c, taut=0.5 fm/c, omega=0.7 c/fm.

## Two different numerical cases

The fresh repeated fine-grid result is **1.450× speedup / 31.04% less wall time**:

| Backend / CPU-side threads | Run 1 (s) | Run 2 (s) | Run 3 (s) | Median (s) |
|---|---:|---:|---:|---:|
| CPU / 4 | 65.798 | 63.958 | 64.617 | 64.617 |
| CPU / 8 | 62.558 | 61.788 | 61.249 | 61.788 |
| CPU / 20 | 83.419 | 82.364 | 84.674 | 83.419 |
| GPU / 8 | 42.914 | 41.962 | 42.610 | 42.610 |

All timed final wavefunctions and printed observables pass the matched CPU
comparison. This is effectively the same performance as the previous 1.441×
checkpoint; it does not establish a significant new acceleration from the
diagnostic change. The timings include startup, diagnostics every ten steps
and the final checkpoint, with density output disabled identically. Jobs run
sequentially after warmups with rotated/reversed order; validation jobs are
separate. Raw timing output: `/home/abhishek/.cache/sky3d-benchmark-3empcs3k`.

The speed benchmark uses the settled 40³/0.6 fm state at dt=0.1 fm/c, 200 steps,
with three sequential interleaved repetitions per CPU/GPU configuration. That
state's long axis is x, so its laboratory M=0 trace does not establish an
intrinsic K=0 spectrum. New timing goes into
`CUDA/results/20ne-40x40x40/energy-fixed/benchmark.json`, preserving old evidence.

The existing 20Ne K=0 spectrum uses the archived z-aligned 24³/1 fm checkpoint,
SLy5, dt=0.2 fm/c, fourth-order propagation, amplitude 5e-5, and a 6000 fm/c
endpoint. Replaying this state is a regression comparison against the existing
calculation; it does not establish fine-grid or static convergence. State hash:
`07353fe3a3fefd74db4a34a129da1f2f969267f9757472193eb6ea3ed5d236f3`.

Both new backends evolve that identical state for 30000 steps, outputting every
2 fm/c. The archived unboosted CPU reference is subtracted identically from
all three spectra. Current CPU/GPU unboosted controls to 200 fm/c check that
reference. Analysis uses the same code-native F20 operator, sign, dt integration
factor, baseline subtraction, zero-padding factor 8 and Gamma_sm=0.5 MeV as the
saved spectrum. No amplitude fit, energy shift or silent interpolation is used.
The transform is first checked by reconstructing the archived spectrum from its
original time signal.

The archived 6000 fm/c CPU checkpoint has neutron/proton counts
9.999996740798872 / 9.999998461853345. Before replay, this measured drift sets an
explicit reproduction-only particle budget of 5e-6. The fine-grid benchmark and
other accuracy checks retain their 1e-6 limit. This is a disclosed regression
budget, not an automatic tolerance change or a production-accuracy claim.

## Full K=0 response comparison

The current parallel CPU replay completed all 30000 steps in 1167.51 seconds
(19.46 minutes). Its full raw quadrupole trace differs from the archived trace
by at most 3.49e-13 fm². Baseline-subtracted relative L2 error is 3.90e-12.
The current CPU strength versus the saved calculation has relative L2 errors
of 9.07e-13, 1.01e-12 and 1.17e-12 for Gamma_sm=1.0, 0.5 and 0.2 MeV.
The transform independently reconstructs the saved CSV curves to relative L2
errors below 1.56e-12. These results preserve the saved response despite the
energy diagnostic correction.

The GPU completed the identical 30000-step input in 1070.31 seconds
(17.84 minutes): 1.091× speedup, or 8.33% less wall time. This is one matched
pair with eight CPU-side threads on both backends, not a repeated benchmark.
The coarse-grid result and repeated fine-grid 1.450× result are different
workloads; neither implies the same speedup for every nucleus or grid.

All full-trace comparisons pass. GPU versus current CPU raw Q differs by at
most 6.04e-13 fm²; GPU versus saved CPU by 5.48e-13 fm². Final CPU/GPU
wavefunctions differ by relative L2 4.03e-12 and maximum absolute 6.77e-13.

| Gamma_sm (MeV) | GPU/current CPU spectrum relative L2 | GPU/saved spectrum relative L2 |
|---:|---:|---:|
| 1.0 | 3.50e-13 | 8.75e-13 |
| 0.5 | 5.22e-13 | 1.06e-12 |
| 0.2 | 8.37e-13 | 1.26e-12 |

These errors cover 0.5–35 MeV. The 10–35 MeV giant-quadrupole region passes
separately. At Gamma_sm=0.5 MeV, current CPU, GPU and saved CPU all put its
largest peak at 16.551539 MeV on the identical energy grid. Peak selection
excludes the separate low-energy peak around 6 MeV.

The summary is `CUDA/results/20ne-k0-response/quadrupole-comparison.json`.
The three PNG/PDF figures in that directory show the time/spectrum overlay,
differences and all three smoothing widths. They were visually reviewed:
the curves overlap, and the difference plots expose the small numerical errors.
Raw input/checkpoints/logs remain in
`/home/abhishek/.cache/sky3d-quadrupole-de8hbe8k`.

## Reproduce

```bash
python3 CUDA/build.py --build-dir "$HOME/.cache/sky3d-energy-fixed-build"
OPENBLAS_NUM_THREADS=1 python3 CUDA/validate_energy_fix.py \
  --state /path/to/settled_fine_20ne.tdhf \
  --build-dir "$HOME/.cache/sky3d-energy-fixed-build" \
  --output CUDA/results/20ne-40x40x40/energy-fix-validation.json
OPENBLAS_NUM_THREADS=1 python3 CUDA/benchmark.py \
  --build-dir "$HOME/.cache/sky3d-energy-fixed-build" \
  --state /path/to/settled_fine_20ne.tdhf --dt 0.1 --initial-flow-atol 1e-5 \
  --mode benchmark --steps 200 --repetitions 3 \
  --output CUDA/results/20ne-40x40x40/energy-fixed
OPENBLAS_NUM_THREADS=1 python3 CUDA/compare_quadrupole.py \
  --build-dir "$HOME/.cache/sky3d-energy-fixed-build" --particle-atol 5e-6 \
  --output CUDA/results/20ne-k0-response
```

The energy-fix regression also requires the previous baseline build in
`CUDA/build`, or `--legacy-build PATH`. The response reproduction requires the
original local static checkpoint and archived response protocols; those large
research outputs are intentionally excluded from Git. Its source script and
small comparison summary/plots are versioned. The 6000 fm/c replay is K=0 only;
K=1/2 and an experimental or external literature comparison need their own
specified inputs and tests.

To regenerate plots from completed cached jobs, add `--run-directory PATH`
and keep the same output report. Reanalysis requires that report, matching
executable hashes, seed hash, completed endpoints and the specified particle
budget. It preserves the original
execution-runner hash and timings while recording the analysis-runner hash.

## Execution status

- Diagnostic correction and pulse/restart/output-interval regression: passed.
- Fresh repeated fine-grid benchmark: passed; 1.450×, 31.04% less wall time.
- Standard, rectangular and reset-CM CPU/GPU controls: passed again.
- Full 6000 fm/c K=0 CPU/GPU/reference time and spectrum comparison: passed;
  all three smoothing widths reproduce the saved local plot.
- Script tests: 18 passed; missing-provenance reanalysis is rejected.
- GPU field construction, diagnostic acceleration and static offload remain
  the later stages of [the single-GPU plan](SINGLE_GPU_PLAN.md).
