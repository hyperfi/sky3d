# Single-GPU implementation and validation

Branch: `gpu`. All compilation, numerical execution and analysis use WSL.
The local hardware is an RTX 5070 with 12 GB VRAM and a Core Ultra 7 265K.
Raw wavefunctions, densities, build products and logs remain in the WSL cache.
The branch contains source, scripts, documentation and small evidence reports.

## Implemented work

The FP64 CUDA C++ backend now includes propagation, spectral derivatives,
all five densities, Skyrme fields, isolated/periodic Coulomb, single-particle
properties and M=0 moments/multipoles. CUDA graphs cover propagation and field
construction. Dynamic densities remain on device through prediction,
averaging and correction. CPU consumers explicitly synchronize current data.
External-field diagnostic and next-predictor time levels remain distinct.

Static preparation uses GPU Hamiltonians, damped gradients, densities and
fields. Ordered orthogonalization, pairing, basis overlaps, small LAPACK
diagonalizations, integrated-energy output and file I/O retain CPU execution.
Other M projections use the native CPU diagnostic formulas. These CPU costs
are included in the complete-job measurements; this is a single-GPU backend,
not a claim that every operation runs on CUDA. Multi-GPU/MPI and out-of-core
execution are deferred.

Static CPU-field fallback also uses CPU moments: its relaxed density is not
uploaded to the device density bank. The regression control caught incorrect
printed geometry when GPU moments used the unrelaxed bank; wavefunctions and
potentials were already correct. The fix preserves the default GPU-field and
all dynamic paths. `static-fallback-before-fix.json` records the failure, and
`static-controls.json` tests the repaired route plus CPU diagnostics/graphs off.

The component controls are `SKY3D_GPU_FIELDS`, `SKY3D_GPU_RESIDENT`,
`SKY3D_GPU_DIAGNOSTICS` and `SKY3D_GPU_GRAPHS`; `0` selects the explicit
comparison path. CPU fields also disable residency. The initial GPU density
bank is populated before diagnostics even when CPU fields are selected.

## Measurements

The field-construction milestone measured 62.2160 s on CPU8 and 30.1976 s on
GPU8: **2.0603x** for 200 steps on the 40³/0.6 fm SLy5 20Ne state, dt=0.1,
Taylor order 4. Three complete jobs per configuration were interleaved;
startup, diagnostics and final checkpoint writing were included. This result
precedes resident densities and diagnostic/static reductions.

The repaired-build 200-step fine-grid medians are CPU4 64.2958 s,
CPU8 63.1325 s, CPU20 83.2065 s and GPU8 **19.8037 s**.
The GPU is **3.1879x faster than the best measured parallel CPU**, with
**68.6316% less wall time**. There are
three uninstrumented complete jobs per configuration, rotated/reversed ordering
and warmups. Startup, identical diagnostics and final checkpoint writing are
included. All final trajectories and printed observables pass their matched
comparisons. This is the exact original x-aligned fine checkpoint, not the
coarse archived K=0 response case. The current report is
`single-gpu/benchmark-final4.json`; the preceding 3.1501x measurement remains
in `single-gpu/benchmark.json` with its original provenance.

The independent 6000 fm/c response replay on the archived z-aligned 24³/1 fm
SLy5 state takes **1206.8849 s on CPU8 and 499.0433 s on GPU8: 2.4184x**.
This is one complete job per backend, rather than the three-repetition
fine-grid performance statistic above. It includes 30000 steps at dt=0.2,
Taylor order 4, matched diagnostics and endpoint checkpoint writing.

CPU, GPU and the saved local CPU calculation have the same GQR peak at
**16.5515393 MeV** (peak selection restricted to 10–35 MeV). GPU versus fresh
CPU strength differs by **5.11e-12 relative L2** at Gamma=0.5 MeV; the largest
raw quadrupole difference is **1.39e-12 fm²**. All three saved smoothing widths
(1, 0.5 and 0.2 MeV) pass without fitting, shifting or rescaling. Endpoint
wavefunctions agree to **2.69e-12 relative L2**. This reproduces the existing
local plot; it is not an independent comparison with experimental data.
The archived coarse-grid state has up to 3.26e-6 particle drift at 6000 fm/c,
so the explicit replay-only budget is 5e-6; the fine-grid budget remains 1e-6.

**The archived coarse response fails the physical orthogonality gate.** At
6000 fm/c both CPU and GPU have an occupied Gram error of 6.4175e-5, above
the unchanged 1e-6 budget. Their Gram matrices differ by only 1.93e-14, and
both have 1.688e-4 MeV maximum printed integrated-energy drift. The replay
passes implementation/spectrum equivalence, but does not qualify this coarse
calculation for production. `single-gpu/response-endpoint.json` deliberately
records `passed=false`; the audit exits 2 rather than relaxing the limit.

A fresh strict CPU-field probe of the archived initial state finds weighted
residual 1.2231e-4 MeV and full projected Hamiltonian antihermiticity up to
4.3389e-4 MeV. The fine benchmark state has residual 1.5982e-8 MeV and
antihermiticity up to 4.5468e-8 MeV. These are different saved states, not a
controlled grid-convergence sequence. The native expanded discrete operator
is retained; no GPU-only symmetrization changes the physics to hide the issue.
See `single-gpu/response-reference-grid-audit.json`. Production response work
needs a reconverged, refined state and long-run grid/timestep checks. The fine
100 fm/c tests and independent-state checks below do not replace that study.

Reports are in `CUDA/results/20ne-40x40x40/single-gpu/` and
`CUDA/results/20ne-k0-final/`, including overlay and difference figures.
Historical hybrid measurements remain in their original directories.

The final isolated 100-step stage profile attributes 2.8621 s to the GPU
predictor and 5.2251 s to the corrector, versus 0.2360 s for midpoint fields
and 0.6507 s for endpoint fields/diagnostics. Propagation is now the dominant
measured cost; field construction and diagnostics no longer account for half
the GPU job. Instrumented and original wavefunctions agree below 1.2e-15
relative L2. These timers locate costs only: inclusive Skyrme/Coulomb/tinfo
timers overlap the exclusive stages, and are not added or used for speed claims.
See `single-gpu/profile.json`.

## Validation and limits

Standard 40³, rectangular 44×42×40 and center-of-mass-reset controls compare
with the strict CPU reference. Dynamic wavefunction limits remain 1e-9 for
relative L2 and absolute error. Saved field limits remain
`1e-9 + 1e-10*max(abs(reference))`. Printed total and integrated energies use
an absolute 2e-7 MeV limit. Particle and occupied Gram checks retain their
documented budgets. Endpoint checks are mandatory: a legacy Fortran `STOP`
can return exit code zero without completing the requested calculation.

One initial rectangular near-vacuum collective-energy comparison differs by
1.1948e-5 MeV between optimized and strict CPU builds, before GPU evaluation.
The explicit rectangular initial-only comparison budget is 2e-5 MeV; all
later collective-energy samples retain 5e-6 MeV. Wavefunction, density, field
and total-energy limits are unchanged. See `initial-flow-cpu-control.json`.
This ill-conditioned `j²/rho` tail is not a precise physical observable.

The final extended 20Ne matrix passes: boosted/unboosted strict-reference
controls at 10 fm/c, dt=0.1/0.05 and Taylor orders 4/6, plus matched CPU/GPU
evolution to 100 fm/c. At 100 fm/c the wavefunction relative L2 difference is
1.91e-14, occupied Gram drift is below 4.96e-9, and printed integrated-energy
drift is 1e-7 MeV on both backends. At 10 fm/c, dt=0.1/order 4 differs from
dt=0.05/order 6 by 5.63e-9 wavefunction relative L2; this is measured short-run
integration sensitivity, not a 6000 fm/c timestep-convergence claim.
See `single-gpu/extended-validation.json` for all 17 numerical jobs.

The broader matrix includes 16O/SLy5 and 20Ne/SLy4 with VDI fractional
occupations, periodic and disabled Coulomb, component fallbacks and nonzero
M controls. Fixed-iteration static controls establish implementation agreement;
independent converged-state preparations additionally check fresh-field
residuals, Hermiticity, occupied subspaces and subsequent TDHF evolution.
Frozen-occupation TDHF is the native approximation here, not TDHFB.

The initial 32³/0.8 fm paired SLy4 CPU preparation exhausted 4000 iterations:
weighted pre-gradient fluctuation 4.46e-6 MeV, fresh direct residual 2.52e-6
MeV and full projected Hamiltonian symmetry error about 2.7e-5 MeV. This is a
failed CPU convergence control, not a passed GPU result. It is preserved in
`independent-nuclei-coarse.json`. The independent-state runner accepts an
explicit finer mesh/spacing and iteration ceiling; it never relaxes `serr`.
The refined 40³/0.6 fm preparation uses the same 1e-6 MeV gate, with an explicit
6000-iteration ceiling. The successful coarse 16O pair remains recorded too.

The paired 40³ control converged but failed the full projected Hermiticity
limit (about 3.8e-7 MeV); 48³/0.5 fm still failed fresh residual/Hermiticity
limits. Their reports are preserved as `independent-nuclei-40.json` and
`paired48.json`. At 56³/0.428571 fm, explicitly tightening the stopping rule
to 1e-7 produces fresh residuals about 1.25e-7 MeV and Hermiticity errors below
1e-9 MeV on both backends. All 36 orbitals have nontrivial occupations.
This is refinement of the CPU reference too; no GPU-specific tolerance is
relaxed. The 56³ dt=0.1/order-4 CPU trajectory became unstable near 4 fm/c;
`paired56-dt01-failure.json` preserves that failure. The validated static
states are reused for an explicit dt=0.05, 200-step comparison at 10 fm/c.
`resume_prepared_dynamics.py` refuses reuse unless the original static and
fresh-field gates pass. Static timings are carried forward with their actual
provenance, rather than presented as new measurements.

The refined paired comparison passes at dt=0.05: CPU 417.6432 s versus GPU
85.5739 s for 10 fm/c, a **4.8805x single-pair result** on this different,
larger workload. Independent static preparation takes CPU 1044.3548 s versus
GPU 581.8827 s (1.7948x), both reaching the stricter criterion at iteration
772. The independent 40³/0.6 fm 16O preparation takes CPU 37.4250 s versus GPU
23.0411 s (1.6243x), both at iteration 261. These static timings include the
retained CPU operations and are not repeated performance statistics.

Both nuclei also pass TDHF started from the GPU's own independently prepared
checkpoint. Density, potentials and printed observables agree with the CPU
preparation/evolution; comparing these physical quantities allows phases and
rotations within degenerate orbital spaces. Frozen occupations remain exact,
particle changes stay below 1e-6 and occupied Gram errors stay below 4.6e-11.
See `independent-nuclei.json`, `own-gpu-16o.json` and `own-gpu-paired56.json`.

The extended and 6000 fm/c response reports precede the static CPU-field
moment repair. That repair changes only the static CPU-field diagnostic
selection; the default dynamic route and CUDA numerical source are unchanged.
Fresh default/fallback controls and uninstrumented timing jobs are repeated on
the repaired build, with each report retaining its actual source/binary hashes.

CUDA memory checking and script tests supplement the numerical checks.
All 19 CUDA script unit tests pass on the final source. Static paired
25-iteration memcheck (including diagonalization) and dynamic memcheck both
report zero errors. The final standard, rectangular and reset controls,
11 component/mode groups and both 10-case static matrices pass.
Agreement with CPU alone does not establish physical production accuracy.
Grid, time-step, force and smoothing choices still belong to the scientific
calculation; larger nuclei and other hardware need their own matched checks.

## VRAM and cloning

For G cells and S propagated orbitals, arrays require
`G*(288*S+1016) + 52*S + ceil(G/256)*max(10*S,38)*8` bytes, plus the maximum
actual cuFFT workspace. This includes the static preconditioner, reductions,
old predictor density and the doubled isolated Coulomb buffers. S includes
empty orbitals. CUDA context, plan/graph metadata and other applications need
additional memory. The runtime refuses allocation above 80% of currently
free VRAM and warns above 60%; preflight also supports memory-pressure
simulations. Capacity alone is not a speed prediction.

Actual queried array/workspace requirements are 0.0873 GiB for 24³/20 states,
0.4043 GiB for 40³/20, 6.2813 GiB for 48³/208 and 14.8889 GiB for 64³/208.
The last case is refused on this 12 GB GPU. Pressure/refusal simulations pass;
a runtime test with an explicit 5% safety fraction exits 3 before allocating
large device banks. See `single-gpu/preflight.json`. These memory fixtures
are not static solutions or performance claims for a 208-nucleon nucleus.

See [CUDA/README.md](../CUDA/README.md) for clone/build/run commands and
component controls. Native binaries must be rebuilt on the destination host.
The `gpu` branch is published; compile it locally after cloning. Static checkpoints
must be prepared locally or transferred separately, not committed to Git.

A clean local clone of checkpoint `c939eb5` rebuilds the reference, CPU and
GPU executables and passes the standard, rectangular and center-of-mass-reset
strict-reference comparisons for 10 steps. The clone remains Git-clean. All
compiled sources match the tested repaired build after LF normalization;
binary equality across different build paths is not claimed. See
`single-gpu/clone-validation.json`. Independent static generation is provided
by the included preparation runner; these clone replay checks deliberately
use the external validated fine seed.

All 337 original local artifacts match their recorded sizes and SHA-256 hashes,
and both protected main refs retain `be42efc7`. See
`single-gpu/preservation-check.json`. The implementation and reproducibility
work is complete. Scientific qualification of the archived coarse long
response remains failed as documented above; it is not promoted to a
production result by these software checks.

## Static alignment and the response frame

TDHF boosts the supplied static state. Alignment names its orientation in
Cartesian coordinates; it does not impose a separate static solution. The
fine checkpoint has mean squares (x,y,z)=(4.42949,2.19509,2.19509) fm², so its
long axis is x. `inspect_orientation.py` reads the density tensor and reports
its eigenvectors without changing the state.

The native lab M=0 quadrupole is proportional to `2z²-x²-y²`. For an axial
nucleus, lab M equals intrinsic K only when the symmetry axis is z. An
x-aligned state can be boosted directly, but that lab operator excites a mix
of intrinsic channels. To compare with the saved K=0 response, use the same
z-aligned state and the same boost/readout frame. For a spherical nucleus
there is no preferred intrinsic axis; a triaxial density has three distinct
principal axes and does not have one axial K label.

The existing `rotate_wf_axis.py` applies active `R_y(-pi/2)` to both spatial
coordinates and spinors. Equal x/z grids permit an exact index permutation,
without interpolation. The fine-state rotation changes discrete norms by at
most 8.1e-15 and interchanges the x/z mean squares. Arbitrary interpolated
rotations require additional grid-level validation. See
`single-gpu/axis-orientation.png`, `orientation.json` and `rotation.json`.
Rotation of deformed TDDFT Slater determinants, including the spin part, is
discussed by [Pigg, Umar and Oberacker](https://arxiv.org/abs/0904.0598).

The fine performance benchmark retains its original x-aligned checkpoint.
The long response replay uses the archived z-aligned 24³/1 fm checkpoint,
matching the existing local plot. These are different workloads and grids;
neither their timings nor their physical spectra should be interchanged.
