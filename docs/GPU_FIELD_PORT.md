# GPU field construction milestone

The single-GPU backend now builds local Skyrme potentials, effective masses,
density derivatives, spin-orbit/time-odd vector fields and Coulomb on the GPU.
The CPU path is retained. `SKY3D_GPU_FIELDS=0` explicitly selects CPU fields in
the GPU-linked executable for comparison/debugging.

The real-field Fourier derivatives retain the CPU collocation convention,
including zero first-derivative Nyquist and retained second-derivative Nyquist.
The expanded effective-mass Hamiltonian is unchanged. The Coulomb solver
uploads the existing CPU-generated kernel once, preserving the discrete origin
value, isolated doubled box and periodic zero-mode convention. Both modes are
implemented; periodic trajectory checks are part of final validation.

GPU initialization now precedes the first field calculation. At this milestone,
host densities and fields are still exchanged at each field call. Removing
those transfers and accelerating diagnostics are the next steps.

The preflight and runtime reserve `G*(288*S+840)+12*S` bytes plus the maximum
actual cuFFT workspace across propagation, real-field and Coulomb plans.
The Coulomb arrays reserve the isolated doubled-box capacity even in periodic
mode. All plans share one workspace on the ordered execution stream.

## Validation

WSL build: `/home/abhishek/.cache/sky3d-field-port-build`.
Raw comparison jobs: `/home/abhishek/.cache/sky3d-benchmark-u6tn3tad`.

Standard 40³, rectangular 44×42×40 and `mrescm=1` cases pass 100-step CPU/GPU
comparisons against the strict CPU reference at dt=0.1 fm/c. Wavefunctions,
all five densities, saved fields, printed observables and initial Hψ/density
checks retain the existing tolerances. Integrated two-step CUDA memcheck passes
with zero errors. Memory queries, pressure simulations and runtime refusal
before oversized allocation pass with the updated formula.

Small reports are in `CUDA/results/20ne-40x40x40/field-port/`.
The individual validation timings suggest improvement, but are not a repeated
speed benchmark. Fresh complete-job timing is recorded separately there.

## Orientation

TDHF can boost any converged static state. X/z alignment describes the density
principal axis in Cartesian coordinates, not a different physical state or a
static solver mode. For an axial nucleus, a lab M=0 quadrupole is intrinsic K=0
only when the symmetry axis is the lab z axis. The finer benchmark's long axis
is x; the saved K=0 response uses the archived exact 90-degree grid/spinor
rotation mapping that axis to z. Boost and readout must use the same frame.
Arbitrary interpolated rotations require their own numerical validation.
