# Compile and run the GPU branch

This guide builds Sky3D for one NVIDIA GPU on Linux or WSL2. The
[oxygen-16 demo](../Test/GPU/README.md) runs from the supplied inputs after
the build. The GPU backend uses FP64 CUDA C++/cuFFT with
GNU Fortran; no NVIDIA Fortran compiler is needed. Multi-GPU, GPU MPI and
out-of-core execution are not implemented.

## 1. Set up the host

The recorded [first-run check](../CUDA/results/20ne-40x40x40/single-gpu/onboarding-validation.json)
used Ubuntu 24.04 under WSL2, CUDA 13.0, gfortran 13.3 and an RTX 5070.

On Windows, install the NVIDIA Windows driver and WSL2. Follow
[NVIDIA's WSL setup guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Install the CUDA **toolkit only** inside WSL, using NVIDIA's WSL-Ubuntu
instructions; do not install a Linux NVIDIA display driver inside WSL.
For native Linux, follow the
[CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
Run the remaining commands in a Linux or WSL2 Bash terminal. On Ubuntu,
install the common build dependencies:

```bash
sudo apt-get update
sudo apt-get install git build-essential gfortran make \
  libfftw3-dev liblapack-dev libopenblas-dev python3
```

Check GPU visibility and the host tools before selecting a toolkit:

```bash
if command -v nvidia-smi >/dev/null; then
  nvidia-smi
else
  /usr/lib/wsl/lib/nvidia-smi
fi
g++ --version
gfortran --version
python3 --version
```

`nvidia-smi` reports the installed driver and the highest CUDA version that
driver supports. `nvcc --version` reports the toolkit compiler version
actually used to build. The commands below invoke the selected toolkit's
`nvcc` by path. [NVIDIA's toolkit, driver and architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)

## 2. Clone and compile

Clone onto the Linux filesystem, for example under `~/src`. Run the build
and demo commands from the repository root.

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone --branch gpu --single-branch https://github.com/hyperfi/sky3d.git
cd sky3d
```

Choose one of the following setups. The builder honors `CUDA_HOME` when set.
With it unset, the builder looks for `nvcc` in `/usr/local/cuda/bin` and then
on `PATH`. The default `--arch native` asks a supporting `nvcc` to detect the
visible GPUs. Use an explicit `sm_*` target when building for another machine,
when the build environment cannot see the target GPU, or when an older
`nvcc` lacks `native` support. NVIDIA added `native` in CUDA 11.5 Update 1.
[NVIDIA nvcc options](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html#gpu-architecture-arch),
[CUDA 11.6.2 release notes](https://docs.nvidia.com/cuda/archive/11.6.2/cuda-toolkit-release-notes/index.html)

### Ubuntu 22.04 distro toolkit with an RTX A2000

Ubuntu 22.04 packages CUDA 11.5 with `/usr/bin/nvcc` and cuFFT in the system
library directories. The toolkit is in Ubuntu's Multiverse repository.
The RTX A2000 has compute capability 8.6, so select
`sm_86` explicitly. [Ubuntu toolkit package](https://packages.ubuntu.com/jammy/nvidia-cuda-toolkit),
[NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus)

```bash
sudo apt-get install nvidia-cuda-toolkit
export CUDA_HOME=/usr
"$CUDA_HOME/bin/nvcc" --version
python3 CUDA/build.py --arch sm_86
```

On the Ubuntu 22.04 workstation, CUDA 11.5 with GCC 11 failed in
`/usr/include/c++/11/bits/std_function.h` with
`error: parameter packs not expanded with '...'`. If you hit this error,
install `g++-10` only if `/usr/bin/g++-10` is missing, then rebuild with
`--host-compiler /usr/bin/g++-10`. GCC 10 is required only for this
workaround. The flag selects it for nvcc and leaves the system compiler
default unchanged. The GCC 10 build and GPU run on that workstation remain
unverified.

```bash
if [ ! -x /usr/bin/g++-10 ]; then
  sudo apt-get install g++-10
fi
python3 CUDA/build.py --arch sm_86 \
  --host-compiler /usr/bin/g++-10
```

### Versioned NVIDIA toolkit

Select the directory for an installed toolkit whose compiler supports the GPU
target and a C++17 host compiler on your Linux distribution. Install the
toolkit's CUDA runtime and cuFFT development files. Check the corresponding
toolkit release notes for driver compatibility and the installation guide for
host compiler support. This example uses CUDA 12.2 and an RTX A2000.
[NVIDIA CUDA 12.2 release notes](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-toolkit-release-notes/index.html),
[CUDA 12.2 Linux installation guide](https://docs.nvidia.com/cuda/archive/12.2.0/cuda-installation-guide-linux/index.html)

```bash
export CUDA_HOME=/usr/local/cuda-12.2
SKY3D_ARCH=sm_86
"$CUDA_HOME/bin/nvcc" --version
"$CUDA_HOME/bin/nvcc" --list-gpu-code | grep -Fx "$SKY3D_ARCH" &&
  python3 CUDA/build.py --arch "$SKY3D_ARCH"
```

### GeForce RTX 5070

The RTX 5070 has compute capability 12.0. Use `--arch sm_120` with CUDA 12.8
or newer. On native Linux, choose an NVIDIA driver with open kernel modules
that lists the RTX 5070 and supports your toolkit release. For example,
Linux x86-64 driver 570.133.07 lists the RTX 5070 and is newer than the
570.124.06 driver listed for CUDA 12.8 Update 1. For a newer toolkit, use
its release notes to select the driver. On WSL2, use a compatible Windows
driver with RTX 5070 support. [NVIDIA compute capability table](https://developer.nvidia.com/cuda/gpus),
[CUDA 12.8 Update 1 release notes](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html),
[Linux driver 570.133.07](https://www.nvidia.com/en-us/drivers/details/242302/),
[NVIDIA open kernel modules guidance](https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/),
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

```bash
export CUDA_HOME=/usr/local/cuda-12.8
"$CUDA_HOME/bin/nvcc" --version
"$CUDA_HOME/bin/nvcc" --list-gpu-code | grep -Fx sm_120 &&
  python3 CUDA/build.py --arch sm_120
```

### Build output

The builder prints the selected `nvcc`, architecture and build directory.
Rebuild locally after moving machines because library paths and native CPU
instructions are host-specific.

The build writes these files under `CUDA/build/`:

| File | Purpose |
| --- | --- |
| `gpu/sky3d.gpu` | GPU-linked static/TDHF executable |
| `cpu/sky3d.cpu` | Optimized parallel CPU comparison |
| `reference/sky3d.reference` | CPU reference without fast-math or FP contraction |
| `libsky3d_gpu.so` | CUDA backend |
| `build.json` | Toolkit and compiler selection, flags, source and executable hashes |

On compiler or linker failure, the builder prints the first diagnostic it
finds and the full path to `cuda-build.log` or the variant's `build.log`.

## 3. Run the demo

After building, follow the [oxygen-16 demo in Test/GPU](../Test/GPU/README.md)
to prepare a ground state and evolve it for 100 steps. The demo uses the
executable in `CUDA/build/` and writes its results under `Test/GPU/`.
To check VRAM before starting,
query the backend arrays and actual cuFFT workspace for its 40³ grid and
16 orbitals:

```bash
python3 CUDA/preflight.py --grid 40 40 40 --states 16
```

## 4. Optional release validation

The validators require **Python 3.11 or newer** with NumPy. On Ubuntu 24.04,
install `python3-numpy` for the system Python 3.12. On Ubuntu 22.04, activate
a separate Python 3.11+ environment with NumPy. For response plots, install
Matplotlib in the same environment.

```bash
mkdir -p CUDA/build/validation "$HOME/.cache"
python3 -m unittest discover -s CUDA/tests -v
python3 CUDA/validate_single_gpu.py --build-dir CUDA/build \
  --mode prepare --prepare-case 16o-sly5 \
  --prepare-mesh 40 --prepare-spacing 0.6 --maxiter 6000 \
  --output CUDA/build/validation/16o.json
python3 CUDA/validate_prepared_tdhf.py --build-dir CUDA/build \
  --preparation CUDA/build/validation/16o.json --case 16o-sly5 \
  --output CUDA/build/validation/16o-own-gpu.json
```

The preparation runner generates CPU and GPU static states independently,
requires the original static stopping criterion, checks fresh CPU-field
residuals, Hamiltonian symmetry and occupied subspaces, then compares short
TDHF runs. The own-preparation check also evolves the GPU's own prepared state.
The matched runs compare GPU output against the CPU baseline using the project's
tolerances, including wavefunction relative L2 and maximum absolute errors
below `1e-9` and an integrated energy tolerance of `2e-7` MeV. The fixture
uses eight CPU threads and evolves 100 steps at 0.1 fm/c.

Success prints `Single-GPU coverage passed` and
`Own GPU-prepared TDHF check passed`. Both reports contain `"passed": true`.
Actual wavefunctions, input files and logs are stored in fresh
directories under `~/.cache/sky3d-*`; their paths are recorded in the JSON.
No external checkpoint download is needed. A failed convergence or comparison
preserves its partial report and raw calculation; inspect the recorded
directory and `stdout.log` before using that state.

## 5. Move to your own nucleus

Prepare and qualify that nucleus's static state first. Use a matching force,
mesh and fragment checkpoint in `for005`; select boost amplitude, multipole,
time step and duration for your scientific calculation. For deformed nuclei,
inspect the density's principal axes: native lab M=0 is about z and need not
be intrinsic K=0. Rotation must include spinors. See the
[alignment discussion](SINGLE_GPU_RESULTS.md#static-alignment-and-the-response-frame).

For the archived 24³/1 fm 20Ne state, a smaller timestep does not remove
the measured overlap error. Prepare and settle a state on the finer mesh
before using it: editing only `&grid` while retaining that coarse checkpoint
does not reproduce the finer-state test. The
[overlap study](ORTHOGONALITY_STUDY.md) gives the controls and remaining
long-response checks.

Use the total propagated state count, including empty orbitals, in preflight.
For one-fragment input it can read that count directly from the checkpoint;
for manual/multiple-fragment cases supply both `--grid` and `--states`.
The backend requires even grid dimensions, `tfft=T`, complex128 and one process.
It warns above 60% of free VRAM and refuses the default 80% safety-budget
limit. Memory fit does not predict speed; use parallel CPU when it refuses.

The specialized `benchmark.py` runner requires SLy5 20Ne with 20 states; the
16O state above is not its input. Static preparation, general Sky3D execution
and preflight support other cases. The [detailed GPU guide](../CUDA/README.md)
contains benchmark, pairing, restart, component controls and memory-check
commands. Its matched paired fixture uses 56³/0.428571 fm and dt=0.05.

The recorded 3.19× result is specific to the tested 20Ne workload/hardware.
The archived coarse 20Ne long response matches CPU/GPU, but fails the physical
orthogonality gate on both. Grid, timestep and long-run scientific convergence
must be assessed separately; see [results and limits](SINGLE_GPU_RESULTS.md).

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `nvcc` missing | Try `command -v nvcc` and `ls -d /usr/local/cuda*`. Set `CUDA_HOME` to the toolkit root containing `bin/nvcc`. |
| `libcufft.so` missing | Install the cuFFT development files for the selected toolkit. |
| Unsupported GPU architecture | Check the selected compiler's `--list-gpu-code` and use an explicit `--arch` supported by your toolkit and GPU. |
| CUDA 11.5 parameter-pack error in `std_function.h` | Install `g++-10` if missing, then rebuild with `--host-compiler /usr/bin/g++-10` as shown above. |
| Missing FFTW/LAPACK/OpenBLAS | Install the development packages above; inspect the variant's `build.log`. |
| Missing `file_digest` or `tomllib` | Use Python 3.11+ in the shell running the scripts. |
| GPU unavailable in WSL | Check the Windows driver and WSL2 setup; run `nvidia-smi` inside WSL. |
| Missing `.so` after moving a build | Rebuild on the destination host; the build contains absolute library paths. |
| Insufficient VRAM | Close competing GPU jobs or use parallel CPU; count all propagated orbitals. |
| No final checkpoint or wrong endpoint | Inspect the calculation log for the expected completion message. |
| Static gate or comparison fails | Preserve the report/raw directory and investigate; do not loosen tolerances automatically. |
| Numerical results differ on another host | Run matched CPU/GPU checks for that host and case before a production calculation. |
