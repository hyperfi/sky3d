# Sky3D GPU feasibility and code audit

Historical audit of the original source. CPU repairs and an integrated CUDA
pilot have since been implemented on `gpu`; see [CPU fixes](CPU_FIXES.md) and
[integrated pilot results](GPU_PILOT_RESULTS.md) for current behavior and timings.

Audit date: 2026-10-08. Physics source: `be42efc`; recorded research workflows:
`6e3ff97`. The Fortran source and production data were not modified.

## Recommendation

Build a single-GPU, FP64 backend for the TDHF propagation and density path,
with state batching and persistent device memory. Start with the existing
gfortran frontend plus a CUDA C++/cuFFT backend through `ISO_C_BINDING`.
This uses the installed toolchain and gives explicit control over memory and
launches. Keep a CPU fallback and the existing file formats.

The local GPU has demonstrated a useful derivative-kernel speedup. It has
**not** demonstrated a whole-Sky3D or production-trajectory speedup. Do not
promise a fixed multiple before integrating and measuring the full step.

## Repository and local machine

- GitHub's repository API confirms `hyperfi/sky3d` is a fork of
  `manybody/sky3d`; the local Git author is `hyperfi`.
- Existing work was committed on `research-workflows` in four focused commits:
  response analysis, generated-file hygiene, 24Mg workflow, and 20Ne workflow.
  `main` remains at `be42efc`. The local `gpu` branch starts from `6e3ff97`.
- No push or upload was performed. No local scientific files were deleted.
  Objects, modules, executables, density dumps, restart states, large HDF5 and
  generated exports remain local and ignored. See `REPOSITORY_HYGIENE.md` and
  `LOCAL_ARTIFACTS.json` for the preservation policy and checksum inventory.
- GPU: NVIDIA GeForce RTX 5070, compute capability 12.0, 12,227 MiB VRAM;
  approximately 10 GB was free during the initial check. Driver: 610.88.
- CPU: Intel Core Ultra 7 265K, 20 cores visible to WSL. WSL exposes about
  15 GiB RAM and 4 GiB swap.
- CUDA 13.0 was already installed at `/usr/local/cuda-13.0`, with
  `/usr/local/cuda` pointing there. `/usr/bin/nvcc` is an older CUDA 12.0.
  WSL `.bashrc` and `.profile` now prefer `/usr/local/cuda/bin`, with backups
  ending in `.cuda-backup-20261008`. New WSL shells resolve `nvcc` to 13.0.
- gfortran 13.3, FFTW and OpenBLAS are available. `nvfortran` is absent.
  Use `/usr/local/cuda/bin/compute-sanitizer`; the distro-installed sanitizer
  and Nsight Systems found on PATH are older. Update the profiler before a
  Blackwell profiling campaign.

NVIDIA introduced native Blackwell toolkit support in CUDA 12.8; the installed
13.0 compiler accepts `sm_120` and successfully executed the local probe.
[NVIDIA's toolkit announcement](https://developer.nvidia.com/blog/cuda-toolkit-12-8-delivers-nvidia-blackwell-support)
documents that support. WSL uses the Windows GPU driver, as described in
[NVIDIA's WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

The RTX 5070's architecture has limited FP64 arithmetic throughput relative
to FP32. That makes batching, memory reuse and FFT efficiency particularly
important; gaming/AI throughput is not an estimate of this code's performance.
Keep double precision for the initial port. The
[RTX Blackwell architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
describes the FP64 limitation and the GB205/5070 architecture.

## Code structure and measured costs

This is Fortran Sky3D v1.2 with CPU FFTW, OpenMP over single-particle states,
optional MPI distribution of states, and LAPACK for static-state work.
There are no CUDA/OpenACC/OpenMP-target kernels in the audited physics path.

The expensive TDHF chain is:

`dynamichf -> tstep -> hpsi -> cdervx/cdervy/cdervz`

Each predictor/corrector stage also calls `add_density`, followed by `skyrme`.
`hpsi` applies first/second spectral derivatives and spin-dependent field
products. `add_density` computes derivatives again and accumulates neutron/
proton number, kinetic, current, spin and spin-orbit densities. Field
derivatives in `skyrme` use explicit matrix contractions, not FFTW, even when
`tfft=T`. Coulomb uses separate 3D FFTs; isolated Coulomb doubles each grid
dimension, increasing its transform volume eightfold.

Source locations:

| Work | Source | Port implication |
|---|---|---|
| State propagation and density reductions | `Code/dynamic.f90:242`, `:269`, `:339` | Batch states; preserve the propagator and field-update order. |
| Hamiltonian | `Code/meanfield.f90:318` | Keep fields/work arrays on GPU; fuse pointwise passes where equivalent. |
| Wave-function derivatives | `Code/levels.f90:115`, `:160`, `:217` | Batched FP64 cuFFT; preserve strides, signs, normalization and Nyquist rules. |
| FFT plans | `Code/fourier.f90:86` | Current y transforms are executed once per z slice and spin; GPU calls must combine work. |
| Density construction | `Code/densities.f90:144` | Replace CPU-sized array reductions with tiled/two-stage GPU reductions. |
| Field construction | `Code/meanfield.f90:124`, `Code/trivial.f90:190` | Port contractions and pointwise fields; a cuFFT relink does not cover these. |
| Coulomb | `Code/coulomb.f90:151` | Preserve isolated Green function and padded-grid treatment. |
| Static solver | `Code/static.f90:217`, `Code/levels.f90:333` | Additional orthogonalization/overlap/diagonalization work; defer to a second phase. |
| Observable output | `Code/dynamic.f90:429`, `Code/moment.f90:124` | Moments are computed every step; keep scalar reductions efficient and retain output conventions. |

A copied serial diagnostic build ran 100 steps of the existing 20Ne K=0
input: 24^3 grid, 20 states, SLy5, no pairing, `dt=0.2`, `mxpact=4`.
The dynamic driver took 11.125 s. Selected **inclusive** times were:

| Routine | Calls | Seconds | Fraction of serial dynamic-driver time |
|---|---:|---:|---:|
| `hpsi` | 12,220 | 7.742 | 69.6% |
| All three `cderv*` routines | 86,040 | 5.727 | 51.5% |
| `add_density` | 4,020 | 1.168 | 10.5% |
| `skyrme` | 201 | 0.994 | 8.9% |
| `tinfo` | 101 | 0.941 | 8.5% |
| `moments` | 101 | 0.688 | 6.2% |
| `poisson` | 201 | 0.286 | 2.6% |

These times overlap: derivatives are inside Hamiltonian/density calls,
moments and diagnostic Hamiltonian calls are inside `tinfo`, and Coulomb is
inside `skyrme`. They must not be added. Serial shares also do not establish
shares after OpenMP parallelization.

Separate complete, uninstrumented 100-step CPU runs were repeated three
times, with interleaved thread counts, one warmup, OpenBLAS restricted to one
thread, and OpenMP core binding. Median elapsed times, including FFT planning,
initialization and output, were:

| OpenMP threads | Median wall time |
|---:|---:|
| 1 | 12.345 s |
| 4 | 5.343 s |
| 8 | 4.594 s |
| 20 | 4.758 s |

Eight threads were slightly better than twenty on this short case. Do not
estimate production duration by simply multiplying these times: startup costs,
long-run CPU scheduling and output frequency matter. The existing flags are
`-O3 -msse4.2 -mfpmath=sse -ffast-math -finline-functions -funroll-loops`.
A fair GPU acceptance benchmark must also consider a tuned native-CPU build,
since this host supports AVX2 beyond the current SSE-targeted flags.

## Actual local GPU probe

`tools/gpu_audit/derivative_probe.cu` computes all three **first** spectral
derivatives in complex FP64. Its CPU side reproduces Sky3D's FFTW layouts and
call granularity with OpenMP across states; it is a standalone C++ harness,
not the unmodified Fortran routine. The GPU packs each axis, batches all
states with cuFFT, applies the spectral multiplier, and unpacks the result.

Both CPU and GPU use reusable plans. The GPU resident timing includes
packing, forward FFT, spectral multiply, inverse FFT, unpacking and final
synchronization. The transfer timing additionally uploads the full input and
downloads all three results using ordinary host memory. Planning/allocation
are excluded. Values are medians of three trials after warmup. CPU and GPU
audit runs were measured sequentially; earlier overlapping trials were
discarded. These are short kernel pilots, not a randomized thermal study.

| Grid / states | CPU, 8 threads | GPU, resident | GPU, with transfers | Resident speedup | Speedup with transfers |
|---|---:|---:|---:|---:|---:|
| 24^3 / 20 | 1.300 ms | 0.650 ms | 3.569 ms | 2.00x | 0.36x |
| 32^3 / 48 | 8.154 ms | 3.434 ms | 19.484 ms | 2.37x | 0.42x |
| 48^3 / 48 | 32.459 ms | 12.641 ms | 60.258 ms | 2.57x | 0.54x |

Relative L2 differences from the FFTW result were below 4e-16; maximum complex
absolute differences were below 3e-15 across the tested shapes. CUDA 13.0
Compute Sanitizer memcheck reported zero errors for 24^3 / 20 states.
This validates this probe's first derivatives and memory accesses. It does
not validate second derivatives, Hamiltonian application, density reduction,
time evolution, static convergence or physical spectra.

The transfer-heavy approach is about 1.9-2.7 times slower than the 8-thread
CPU harness for these batched cases. A wrapper that copies arrays for each
derivative is therefore a poor starting architecture on this machine.
[cuFFT documentation](https://docs.nvidia.com/cuda/cufft/index.html) covers
FP64, batching, strided transforms, device memory and reusable plans.

Even in the serial diagnostic, accelerating a 51.5% derivative share by 2x
would imply only about 1.35x overall by Amdahl's law, before new overheads.
That is an illustration, not a prediction for the threaded code. To obtain
a substantial whole-run gain, move Hamiltonian products, propagation and
density accumulation together and measure the remaining field/output work.

## Implementation options

| Option | Approach | Suitability on this machine | Main cost |
|---|---|---|---|
| **1. gfortran + CUDA C++/cuFFT (recommended)** | Keep Fortran orchestration/I/O; hold an opaque GPU state through `ISO_C_BINDING`; perform batched Hamiltonian/propagation/density work in CUDA. | CUDA 13.0 and gfortran already work; probe demonstrates the numerical primitive. | Explicit API, array-layout and reduction work; persistent ownership must be designed. |
| **2. OpenACC + cuFFT** | Add persistent data regions and loop directives in Fortran; use device pointers for cuFFT; replace large state reductions. | Preserves more Fortran source; appropriate if keeping one language is the priority. | Install compatible NVIDIA HPC SDK; nested routines, module pointers, array expressions and library calls need real porting. |
| **3. CUDA Fortran + cuFFT** | Write explicit batched GPU kernels and device arrays in Fortran. | Good Fortran-first control over kernels; useful alternative to option 1. | Install HPC SDK, maintain a compiler-specific GPU path, and restructure scratch/layouts. |

NVIDIA documents OpenACC/CUDA Fortran support and cuFFT interfaces in its
[HPC SDK overview](https://developer.nvidia.com/hpc-sdk) and
[CUDA Fortran guide](https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/index.html).
Blackwell `cc120` support was introduced in
[HPC SDK 25.3](https://docs.nvidia.com/hpc-sdk/archive/25.3/hpc-sdk-release-notes/index.html).
SDK execution in this particular WSL setup has not yet been tested.

For all three options, the port should:

1. Allocate GPU state, fields, scratch and plans once; retain wave functions
   between steps. Use short, explicit transfers for field boundaries during
   an initial hybrid stage, rather than per-FFT transfers.
2. Batch across occupied states and both spin components; use tiled state
   batches for large nuclei. Preserve FP64 from input through reductions.
3. Port `hpsi`, the Taylor recurrence and `add_density` as a coherent unit.
   Keep the polynomial order, time step, field-update sequence, and operators.
4. Measure the hybrid stage, then port `skyrme`, real-field contractions and
   Coulomb where the remaining end-to-end profile justifies them.
5. Retain existing observables, snapshot/restart cadence and binary formats;
   download scalars or requested snapshots rather than whole states each step.

An initial GPU API should expose state creation/destruction and a whole
propagation stage or step. A Fortran wrapper around every tiny FFT call
would preserve the current launch fragmentation and transfer penalties.

Wave-function storage alone is `32 * nx * ny * nz * states` bytes. Six full
FP64 spinor banks would use about 51 MiB for 24^3/20, 972 MiB for 48^3/48,
and 9.75 GiB for 64^3/208. These are design estimates, excluding fields,
Coulomb buffers and cuFFT workspaces. The largest example needs careful
workspace accounting and tiled scratch on a 12-GB display GPU.

Avoid starting with Coulomb alone (small measured serial share), the small
static eigensolve, Python spectrum postprocessing, an FFTW-to-cuFFTW relink
with ordinary host arrays, or low-precision Tensor Core arithmetic. None
has demonstrated a production TDHF speedup here. Static orthogonalization
and overlap work can be considered after the TDHF path is validated.

## Correctness findings

The following findings were kept separate from the port. No physics fixes
were applied during this audit.

### P1: optional center-of-mass reset doubles densities

`Code/dynamic.f90:283-295` calls `resetcm` and then adds corrected densities
to the already populated density arrays. `resetcm` changes wave-function
phases but does not clear those arrays. In an isolated two-step 20Ne test,
the baseline stays at 10 neutrons / 10 protons; with `mrescm=1`, printed counts
become 20 / 20 after the first step and remain doubled after the second.
The fields built from those densities are therefore affected.

Clear all five density arrays before reaccumulating after the reset, then
verify particle number, current and energy behavior. The saved 20Ne and
24Mg inputs use the default `mrescm=0`, so this finding does not establish
that those earlier calculations were affected.

### P2: disabling restart output with zero interval crashes

`Code/params.f90:95` defaults `mrest` to zero, and the documentation says
restart output is enabled when the value is positive. Dynamic output at
`Code/dynamic.f90:303` uses `MOD(iter,mrest)` without guarding zero.
The isolated two-step `mrest=0` test terminates with SIGFPE (return code -8).
The static loop also contains an unguarded restart modulo at
`Code/static.f90:328`; that path was identified by inspection, not reproduced.
Use a nested positive-interval check before evaluating MOD. `mplot=0` was
tested separately and works because its dynamic check is guarded.

### P2: fragment paths silently truncate at 64 characters

`Code/fragments.f90:48` declares `CHARACTER(64) :: filename(mnof)`.
The first audit input used an absolute state path; truncation left a directory
name and the run failed while opening the fragment. A short symlink fixed
the audit input. Expand/validate path capacity before accepting long paths;
keep restart compatibility in mind if changing serialized strings elsewhere.

### Performance and reproducibility issues to address during the port

- `hpsi`, `add_density`, `skyrme` and `poisson` repeatedly allocate scratch
  arrays. GPU equivalents should use reusable workspaces. CPU scratch must
  remain thread-private if optimized.
- Large OpenMP density-array reductions and repeated array-expression passes
  need a different GPU accumulation design; unrestricted FP64 atomics are
  not a default performance solution.
- The current CPU build uses `-ffast-math`. Establish a strict reference
  alongside the fast baseline for NaN/vacuum behavior, energy drift and
  long-time comparisons. Do not infer numerical equivalence from a build.
- Do not change periodic spectral wave-function derivatives when preserving
  `periodic=F`: that flag changes the Coulomb treatment, not an absorber.
- Preserve Sky3D's negative boost phase, multipole/readout normalization,
  signed cross response and positive-M tesseral convention.

## Gate for claiming a real speedup

First validate primitives: first/second derivatives, `hpsi`, all five density
arrays, Coulomb including isolated padding, and a full time step. Compare
norm/overlap, neutron/proton number, densities, energy and moments against a
strict CPU reference, with stated absolute and relative tolerances.

Then run matched no-boost and boosted trajectories at unchanged settings;
check drift and response linearity, signed spectra and peak/strength changes.
Use the existing 20Ne K=0/1/2 workflow and 24Mg response cases as scientific
regressions; a 20-fm/c timing probe cannot validate a 6000-fm/c spectrum.

Finally compare complete uninstrumented CPU and GPU jobs with identical
initial states, grid, step count, output cadence and physics. Tune CPU thread
count and native compiler options; include device initialization, transfers,
reductions and requested I/O. Warm up, interleave at least three repetitions,
record medians/spread and validate another grid/nucleus. A practical first
acceptance target is at least 1.5x complete-run speedup over the best validated
local CPU baseline, with unchanged physical accuracy. This is a proposed
criterion, not an achieved result.

Evidence and reproduction commands are in `tools/gpu_audit/README.md` and its
small `evidence/` JSON/CSV files. All audit executables, snapshots and raw logs
are retained locally outside Git.
