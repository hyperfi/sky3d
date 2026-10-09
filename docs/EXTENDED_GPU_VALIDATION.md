# First single-GPU validation milestone

The initial block of [the single-GPU plan](SINGLE_GPU_PLAN.md) completed on
2026-10-09 in WSL using the existing compiled hybrid implementation. This
checkpoint adds validation and a measured diagnostic finding; it adds no new
production kernels or changes to the propagator/physics operator.

The state remains the settled Sly5 20Ne checkpoint on a 40³/0.6 fm mesh, SHA-256
`ea7ec8de58dcbe1d37a3c003cffa263b1a58542af27d3abc9f99078c5cea5abc`.
The boost is the laboratory L=2, M=0 pulse with amplitude 5e-5; this state is
elongated along x, so these results do not establish an intrinsic K=0 spectrum.

## Fixed-setting equivalence

All **17 numerical jobs** passed their declared endpoint, finite-output,
particle, occupied-overlap and CPU/GPU comparison checks. The five short
controls each run strict one-thread CPU, optimized eight-thread CPU and GPU
with eight CPU-side threads. The longer case compares optimized CPU with GPU.

| Case | Endpoint, fm/c | dt, fm/c | Taylor order | Maximum GPU wavefunction difference from the case's CPU reference |
|---|---:|---:|---:|---:|
| Boosted | 10 | 0.1 | 4 | 2.80e-14 |
| Boosted | 10 | 0.05 | 4 | 2.80e-14 |
| Boosted | 10 | 0.1 | 6 | 2.80e-14 |
| Boosted | 10 | 0.05 | 6 | 2.80e-14 |
| Unboosted | 10 | 0.1 | 4 | 2.79e-14 |
| Longer boosted run | 100 | 0.1 | 4 | 5.59e-16 |

The longer run reached step 1000 on both backends. Its relative wavefunction
difference is 3.99e-15. GPU neutron/proton checkpoint counts are
9.999999998344336 / 9.999999999245173; maximum count error is 1.66e-9.
The maximum final occupied-state Gram-matrix error is 4.95e-9, versus the
explicit 1e-6 check. Overlap drift is measured relative to the supplied static
checkpoint, including initialization. Neutron/proton overlaps are checked
separately with the physical cell volume.

Final densities/fields and complete printed energy, monopole and quadrupole
histories also pass the existing equivalence tolerances. The explicit
time-zero near-vacuum flow tolerance remains 1e-5 MeV, with 5e-6 at later
times; total/integrated energies retain their 2e-7 MeV comparison limit.
No tolerance was automatically changed during these runs.

The short runs use diagnostics every 1 fm/c, including matching physical
output times across different timesteps. The raw output is in
`/home/abhishek/.cache/sky3d-extended-r1hcrfb7`. Executable/library/source and
input/state hashes, individual comparisons and conservation measurements are
in `CUDA/results/20ne-40x40x40/extended-validation.json`.

## Integration sensitivity

Changing timestep/order is measured separately from backend equivalence.
Relative wavefunction differences at 10 fm/c against strict CPU with dt=0.05,
order 6 are 5.63e-9 for dt=0.1/order 4, 3.28e-10 for dt=0.05/order 4, and
4.21e-9 for dt=0.1/order 6. These are sensitivity measurements, not an assigned
production-accuracy pass or a recommendation to change the Taylor order.

The unboosted printed integrated energy changes by 1e-7 MeV over 10 fm/c.
The longer boosted run's maximum reported integrated-energy departure from
its initial value is 2.7e-6 MeV. Interpreting those numbers requires the
diagnostic time-level check below.

## Measured diagnostic timing issue

`Code/dynamic.f90` calls `tinfo` after advancing time and before the final
`skyrme` refresh. `Code/energies.f90:integ_energy` combines the current proton
density with the preceding `wcoul` in its direct Coulomb energy. The saved
diagnostic Coulomb field therefore need not match the saved endpoint density.
This affects both CPU and GPU; their mutual agreement does not catch it.

`CUDA/check_energy_timing.py` independently reproduces the implemented
isolated double-grid Coulomb convolution. It uses the actual kernel origin
`2.84*3/(dx+dy+dz)` and e²=1.43989, validates analytic point-charge and boundary
cases, and agrees with Sky3D's time-zero potential to 2.67e-15 MeV across all
17 jobs. Then it recomputes the field from each saved endpoint density.

For the 100 fm/c GPU endpoint, the field difference reaches 9.48e-7 MeV and
the direct Coulomb energy correction is +1.3543e-6 MeV. Applying that correction
to the printed endpoint energy change of -1.3e-6 MeV leaves +5.43e-8 MeV,
within the 1e-7 MeV print precision. The shorter boosted endpoints show the
same effect. This isolates a diagnostic contribution; it does not repair the
output, recompute the single-particle energy, or establish exact conservation.
The measurements are in `CUDA/results/20ne-40x40x40/energy-timing.json`.

**Next correctness task:** resolve the energy diagnostic time level while
preserving predictor/corrector and time-dependent external-field semantics.
Test fresh-field energy output, diagnostic intervals and unchanged trajectories
on both backends before advancing to the field port. Do not change the
Hamiltonian discretization to compensate for a diagnostic issue.

## Remaining coverage

This completes the initial 20Ne control/integration/longer-run block. Stage 1
still needs the diagnostic correction, additional nuclei/forces/occupations,
longer restart histories, diagnostic interval controls and explicit physical
accuracy criteria. A 100 fm/c evolution does not validate a production response
spectrum. Stages 2–5 remain pending; multi-GPU stays deferred.

The 15 script tests pass in WSL. Numerical runs contain validation and density
output and are not used for a new performance claim. The previous measured
1.441× speedup remains the baseline for the unchanged compiled implementation.
All 337 preserved local artifacts still match their recorded hashes. Git
contains the scripts, plan, this report and small JSON summaries; raw state,
density, log and compiler output remain local.
