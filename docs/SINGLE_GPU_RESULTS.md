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

Final repeated timing and the production-length quadrupole replay are being
recorded separately in `CUDA/results/20ne-40x40x40/single-gpu/` and
`CUDA/results/20ne-k0-final/`. Historical hybrid measurements remain in their
original directories; they must not be described as current measurements.

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

The broader matrix includes 16O/SLy5 and 20Ne/SLy4 with VDI fractional
occupations, periodic and disabled Coulomb, component fallbacks and nonzero
M controls. Fixed-iteration static controls establish implementation agreement;
independent converged-state preparations additionally check fresh-field
residuals, Hermiticity, occupied subspaces and subsequent TDHF evolution.
Frozen-occupation TDHF is the native approximation here, not TDHFB.

CUDA memory checking and script tests supplement the numerical checks.
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

See [CUDA/README.md](../CUDA/README.md) for clone/build/run commands and
component controls. Native binaries must be rebuilt on the destination host.
The branch is kept local until publication is requested. Static checkpoints
must be prepared locally or transferred separately, not committed to Git.

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
