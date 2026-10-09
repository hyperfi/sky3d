# CUDA C++ TDHF pilot

This branch keeps Sky3D's Fortran input, fields and output and uses a C ABI to
call an FP64 CUDA C++/cuFFT backend. The time-dependent predictor/corrector,
Hamiltonian application, Fourier derivatives, and five density calculations
run on one GPU. Wavefunctions, Taylor buffers and FFT workspace stay resident.
Field construction (including Coulomb), moments, diagnostic Hamiltonian calls,
static relaxation and file I/O still run on the CPU. This is an integrated
pilot, not a complete GPU rewrite.

The CPU control fixes are described in [CPU_FIXES.md](../docs/CPU_FIXES.md).
The historical [GPU audit](../docs/GPU_AUDIT.md) contains measurements of the
earlier source and standalone derivative probe; use the results here for this
integrated implementation.

## Build in WSL or Linux

Requirements: a compatible NVIDIA GPU/driver, CUDA toolkit with cuFFT,
g++/gfortran, make, FFTW3, LAPACK/OpenBLAS, Python 3.10+ and NumPy.
This machine uses CUDA 13.0, gfortran 13.3, an RTX 5070 with 12 GB VRAM,
and an Intel Core Ultra 7 265K. CUDA 13.0's nvcc is now the local WSL default.
No NVIDIA Fortran compiler is required.

```bash
cd /mnt/d/Coding/sky3d
export CUDA_HOME=/usr/local/cuda
python3 CUDA/build.py
```

The builder creates isolated `reference`, `cpu`, and `gpu` executables and
`libsky3d_gpu.so` under the ignored `CUDA/build/` directory. `Code/` objects and
production runs are untouched. The reference disables fast-math and FP
contraction. Both performance builds use `-O3 -march=native -ffast-math
-fopenmp` for their CPU code; CUDA uses complex128 without `--use_fast_math`.
`build.json` records flags, compiler versions, source and executable hashes.
Rebuild on each new machine: native CPU/GPU binaries and their library paths
are local artifacts. `--arch sm_120` can explicitly target this GPU; the
default `native` targets the installed GPU.

The normal `Code/Makefile` and the 24Mg project Makefile still build CPU-only
executables without CUDA. Those executables reject `SKY3D_BACKEND=gpu`.

## Run and check VRAM

Use a fresh calculation directory containing `for005`, with fragment paths
resolved relative to that directory. This pilot requires `tfft=T`, even grid
dimensions, one process, and a complex128 Fortran build. It rejects MPI GPU
mode; this branch has no multi-GPU decomposition or out-of-core mode.

```bash
# From the repository, query a planned grid and ALL propagated orbitals:
python3 CUDA/preflight.py --grid 24 24 24 --states 20

# Or use a one-fragment for005 input and read nstmax from its checkpoint:
python3 CUDA/preflight.py --input /path/to/calculation/for005

# From the calculation directory:
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close OMP_PLACES=cores
SKY3D_BACKEND=gpu /mnt/d/Coding/sky3d/CUDA/build/gpu/sky3d.gpu
# CPU comparison, with the same input, in another fresh directory:
SKY3D_BACKEND=cpu /mnt/d/Coding/sky3d/CUDA/build/cpu/sky3d.cpu
```

`SKY3D_BACKEND=cpu` also selects CPU execution in the GPU-linked executable.
Without this variable, the GPU-linked executable defaults to GPU in dynamic
mode; static mode remains CPU. Device 0 is used, respecting CUDA's device
visibility mapping. Set `CUDA_VISIBLE_DEVICES` before starting to select a GPU.

The preflight uses the backend's allocation formula and **actual cuFFT plan
workspace**, queried without allocating transform arrays. It accounts for
all propagated states, including unoccupied orbitals; mass number alone is
not a sufficient memory estimate. For G grid cells and S states the arrays
require `G * (288*S + 400) + 12*S` bytes, plus cuFFT workspace. The CUDA
context, plans, graph metadata and other applications also consume memory;
the safety reserve is therefore necessary. Host RAM is needed independently
for Sky3D's normal arrays and CPU work buffers.

The runtime independently checks currently free VRAM before array allocation.
It refuses cases above 80% of free memory and warns above 60%. Allocation can
still fail if another application consumes memory after the check. The optional
`SKY3D_GPU_MEMORY_FRACTION` permits values 0.05–0.95; the default 0.80 is
recommended. Use `--safety-fraction` to match an override in the CLI.
Exit codes are 0 for fit/pressure, 2 for insufficient memory, and 1 for an
unavailable/invalid preflight. A pressure result prints a warning.

```bash
# Simulate a tighter memory budget, while querying real cuFFT workspace:
python3 CUDA/preflight.py --grid 48 48 48 --states 208 --available-gib 6
```

Simulations are labeled and do not run the nucleus. Memory fit cannot predict
runtime. Oversubscription or extra chunk transfers in an alternative backend
may make it slower than parallel CPU; this backend refuses oversubscription.
Benchmark on the target machine before choosing GPU for a larger nucleus.

CUDA graphs replay each fixed propagation sequence, reducing repeated host
launches. Each Taylor order, timestep and alternating wavefunction pointer
has a separate cached graph. Field values are uploaded in place between
stages. `SKY3D_GPU_GRAPHS=0` disables replay for diagnosis or comparison.
The implementation uses one nonblocking stream and shared workspace, so FFTs
and buffer reuse remain ordered. See NVIDIA's [cuFFT graph support](https://docs.nvidia.com/cuda/cufft/index.html#cuda-graphs-support)
and [CUDA graphs guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

## Reproduce the small-nucleus checks

Large scientific data are excluded from Git. A fresh clone can generate a
20Ne state from the included SLy5 static input:

```bash
python3 CUDA/benchmark.py --prepare-state --static-serr 1e-4
```

The static calculation runs on CPU with the input's 3000-iteration limit.
The original 24³ static input did not reach its `serr=1e-6` criterion locally;
its measured fluctuation plateau was about `7.56e-5`. The command above
**explicitly** chooses `1e-4` for generating a performance benchmark state.
The production input remains unchanged, and the script never relaxes a
tolerance automatically. Omit the override to demand the original criterion,
or provide a separately validated checkpoint with `--state`.
State preparation is excluded from TDHF timings. The runner checks the
requested criterion and records it in `results/preparation.json`. A newly prepared
orientation can differ from the local z-aligned state used in saved results;
all CPU/GPU variants within one benchmark use the exact same state and hash.
These are implementation checks, not a claim of a new physical spectrum.

To reproduce with a local checkpoint:

```bash
python3 CUDA/benchmark.py \
  --state projects/icnpa2026_20Ne_Kresolved_E2/static/20ne_sly5_zaxis.tdhf
python3 CUDA/benchmark.py --state /path/to/20ne.tdhf --mode sanitizer
python3 CUDA/benchmark.py --state /path/to/20ne.tdhf --mode restart
python3 CUDA/check_preflight.py
python3 CUDA/check_static.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s CUDA/tests -v
```

The default validation runs 100 steps for 24³ and rectangular 28×26×24 grids,
and for `mrescm=1`. It compares the final complex wavefunctions, neutron/proton
rho/tau/current/spin/spin-orbit densities, potentials, printed energies and
moments against a one-thread strict CPU reference. It also compares initial
Hamiltonian and densities directly. Near-zero vector fields use absolute
tolerances. Tolerances and measured errors are in `results/validation.json`.
The near-zero collective-flow diagnostic (`j²/rho`) is sensitive to roundoff
in rectangular padding: CPU fast-math versus strict CPU alone differed by
about `1.4e-6 MeV` at time zero. Its absolute comparison tolerance is `5e-6 MeV`;
total/integrated/kinetic energies and moments retain the tighter tolerance.
The sanitizer checks an integrated two-step GPU run separately from timings.
The restart check resumes the same strict CPU checkpoint from step 20 to 40
on all three executables and compares final wavefunctions, fields and output.

The default benchmark uses 500 steps (`dt=0.2 fm/c`, 100 fm/c), three complete
jobs per configuration, CPU thread counts 4/8/20 and GPU CPU-side threads 8.
Jobs run sequentially with rotated/reversed ordering and separate warmups.
Startup, initial/final checkpoints and `mprint=10` diagnostic work are included;
`mplot=0` is identical for all timing jobs. The runner checks finite outputs,
particle numbers and all final GPU wavefunctions/printed trajectories against
the best measured CPU configuration. Validation overhead and sanitizer are
excluded. Adjust `--cpu-threads` and `--gpu-threads` for another machine.

Raw inputs, logs, checkpoints and density files stay in a fresh directory
under `~/.cache/sky3d-benchmark-*`. Only small JSON summaries are versioned.
`results/benchmark-launches.json` preserves the pre-graph integrated benchmark;
`results/benchmark.json` contains the final implementation's timings.

The local result and numerical limits are summarized in
[GPU_PILOT_RESULTS.md](../docs/GPU_PILOT_RESULTS.md). Larger nuclei, long physical
trajectories, other forces/pairing options, longer restart histories and different
machines need their own comparisons before production use.

The follow-up [static convergence study](../docs/STATIC_CONVERGENCE.md)
explains the coarse-grid plateau and provides a settled 40³/0.6 fm CPU state
meeting the original `1e-6` threshold in fresh fields. Reproduce it with
`static_convergence.py`; its diagnostic builds and raw output stay in WSL's
cache. This finer checkpoint has separate GPU validation and matched timing;
`benchmark.py` now reads grid dimensions and spacing from the checkpoint.
For a non-24³ grid, results default to a separate directory such as
`results/20ne-40x40x40/`, preserving the earlier 24³ evidence.

The 40³ mesh requires a smaller tested timestep: `dt=0.2` became unstable
even on the strict CPU reference. The finer-grid comparisons use an explicit
`dt=0.1`, with Taylor order 4. The time-zero near-vacuum flow diagnostic also
needs an explicit `1e-5 MeV` tolerance: CPU fast-math versus strict CPU differed
by `7.9e-6 MeV` in rectangular padding. Later flow values retain `5e-6 MeV`,
and wavefunctions, fields and total energies keep their existing tolerances.

```bash
python3 CUDA/benchmark.py --state /path/to/fine_20ne.tdhf \
  --dt 0.1 --initial-flow-atol 1e-5 --mode validate
python3 CUDA/benchmark.py --state /path/to/fine_20ne.tdhf \
  --dt 0.1 --initial-flow-atol 1e-5 --mode benchmark \
  --steps 200 --repetitions 3
python3 CUDA/benchmark.py --state /path/to/fine_20ne.tdhf \
  --dt 0.1 --initial-flow-atol 1e-5 --mode sanitizer
python3 CUDA/benchmark.py --state /path/to/fine_20ne.tdhf \
  --dt 0.1 --initial-flow-atol 1e-5 --mode restart
python3 CUDA/profile_stages.py --state /path/to/fine_20ne.tdhf \
  --dt 0.1 --steps 100 \
  --validation CUDA/results/20ne-40x40x40/validation.json \
  --output CUDA/results/20ne-40x40x40/profile.json
```

The stage profiler builds instrumented copies in WSL's cache, verifies their
wavefunctions and observables against passed uninstrumented validation jobs,
and times the serialized coordinator around OpenMP/CUDA work. Its GPU stage
times include synchronization and density download; field stages include
upload. Nested Skyrme, Coulomb and diagnostic timings must not be added to
the non-overlapping stage totals. Use separate uninstrumented jobs for speedup.

The [fine-grid checkpoint report](../docs/FINE_GRID_GPU_RESULTS.md) records
passed validation, memory checking and restart comparisons, followed by three
200-step timing repetitions. The local medians are 62.60 seconds for the best
CPU configuration and 43.45 seconds for GPU (1.441×, 30.6% less wall time).
CPU field construction and diagnostics account for about half the instrumented
GPU job and are the next measured optimization targets.

The [single-GPU plan](../docs/SINGLE_GPU_PLAN.md) sets the order and numerical
gates for further work; multi-GPU is deferred. Its first validation block uses
the settled Sly5 20Ne state, runs 10 fm/c boosted/unboosted controls against
strict CPU, exercises dt=0.1/0.05 and Taylor orders 4/6, and extends the matched
parallel CPU/GPU trajectory to 100 fm/c:

```bash
OPENBLAS_NUM_THREADS=1 python3 CUDA/validate_extended.py \
  --state /path/to/fine_20ne.tdhf --initial-flow-atol 1e-5 \
  --output CUDA/results/20ne-40x40x40/extended-validation.json
```

The runner verifies source/executable/library hashes against the build
manifest. It checks endpoint, particle count, occupied-state overlaps and
fixed-setting CPU/GPU equivalence, with raw output in a fresh WSL cache
directory. The longer run compares optimized parallel CPU with GPU; strict CPU
comparisons are the short controls. Timestep/order sensitivity and integrated
energy drift are recorded separately, without assigning them a production
accuracy pass. The JSON preserves partial evidence on a failure. This block
does not complete the wider validation matrix or establish a response spectrum.

Audit the direct Coulomb term's diagnostic time level after that report passes:

```bash
OPENBLAS_NUM_THREADS=1 python3 CUDA/check_energy_timing.py \
  --validation CUDA/results/20ne-40x40x40/extended-validation.json \
  --output CUDA/results/20ne-40x40x40/energy-timing.json
```

The current `tinfo` call precedes the final `skyrme` field refresh. The audit
recomputes isolated Coulomb from each saved endpoint density using the existing
solver's kernel, verifies its time-zero potential against Sky3D, and measures
the direct Coulomb energy correction. It changes no evolution or output files.
Printed energies are rounded to `1e-7 MeV`; this check does not assign a physical
convergence pass or correct the single-particle diagnostic energy.

The [first validation milestone](../docs/EXTENDED_GPU_VALIDATION.md) records
17 passed numerical jobs and the measured diagnostic timing issue. Stage 1 is
still in progress; the next correctness task is to resolve that time level
before adding the field acceleration.
