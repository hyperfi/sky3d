# Compile and run the GPU branch

This guide starts from source and prepares its own small 16O state. It covers
one NVIDIA GPU on Linux or WSL2. The GPU backend uses FP64 CUDA C++/cuFFT with
GNU Fortran; no NVIDIA Fortran compiler is needed. Multi-GPU, GPU MPI and
out-of-core execution are not implemented.

## 1. Set up the host

The tested host uses Ubuntu 24.04 under WSL2, CUDA 13.0, gfortran 13.3 and an
RTX 5070. Other hosts need a toolkit and driver compatible with their GPU;
the same performance or numerical qualification is not assumed.

On Windows, install the NVIDIA Windows driver and WSL2. Follow
[NVIDIA's WSL setup guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).
Install the CUDA **toolkit only** inside WSL, using NVIDIA's WSL-Ubuntu
instructions; do not install a Linux NVIDIA display driver inside WSL.
For native Linux, follow the
[CUDA Linux installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
Use a toolkit supporting your architecture; RTX 5070 needs CUDA 12.8 or newer.
CUDA 13.0 is the version tested for this branch.

All remaining commands run in a Linux/WSL bash terminal. On Ubuntu 24.04,
install the non-CUDA dependencies:

```bash
sudo apt-get update
sudo apt-get install git build-essential gfortran make \
  libfftw3-dev liblapack-dev libopenblas-dev python3 python3-numpy
```

Python must be **3.11 or newer**: the validators use `hashlib.file_digest` and
the response package uses `tomllib`. Ubuntu 24.04 supplies Python 3.12.
For response plots, also install `python3-matplotlib`; it is optional for the
first calculation below. See [Python's file hashing documentation](https://docs.python.org/3/library/hashlib.html#hashlib.file_digest).

Check the driver and actual compiler before building. Set `CUDA_HOME` to your
toolkit installation; `/usr/local/cuda` is the usual location, and a versioned
location such as `/usr/local/cuda-13.0` also works.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
nvidia-smi
nvcc --version
gfortran --version
python3 --version
python3 -c 'import numpy; print(numpy.__version__)'
```

The CUDA version displayed by `nvidia-smi` is a driver capability; it does not
prove that `nvcc` or the toolkit is installed. In WSL, `nvidia-smi` is also
available at `/usr/lib/wsl/lib/nvidia-smi` if it is missing from `PATH`.

## 2. Clone and compile

Clone onto the Linux filesystem, for example under `~/src`, and keep the
following variables in the same terminal session:

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"
git clone --branch gpu --single-branch https://github.com/hyperfi/sky3d.git
cd sky3d
export SKY3D_REPO="$PWD"
export SKY3D_BUILD="$HOME/.cache/sky3d-gpu-build"
export SKY3D_RESULTS="$HOME/.cache/sky3d-first-run"
mkdir -p "$SKY3D_RESULTS"
python3 CUDA/build.py --build-dir "$SKY3D_BUILD" --arch native
```

`native` targets the visible local GPU. To specify an architecture explicitly,
use `--arch sm_120` for an RTX 5070, or the appropriate architecture supported
by your toolkit. Rebuild locally after moving machines; executable/library
paths and native CPU instructions are host-specific.

The build produces:

| File under `SKY3D_BUILD` | Purpose |
| --- | --- |
| `gpu/sky3d.gpu` | GPU-linked static/TDHF executable |
| `cpu/sky3d.cpu` | Optimized parallel CPU comparison |
| `reference/sky3d.reference` | CPU reference without fast-math or FP contraction |
| `libsky3d_gpu.so` | CUDA backend |
| `build.json` | Compiler, flags, source and executable hashes |

Build errors are saved in `cuda-build.log` or the variant's `build.log`.
The builder leaves the original `Code/` objects and scientific data alone.

## 3. Check VRAM, then prepare and validate 16O

This fixture propagates 16 orbitals on a 40³ grid with 0.6 fm spacing.
The preflight queries actual cuFFT workspace as well as the backend arrays:

```bash
python3 CUDA/preflight.py --library "$SKY3D_BUILD/libsky3d_gpu.so" \
  --grid 40 40 40 --states 16
python3 -m unittest discover -s CUDA/tests -v
python3 CUDA/validate_single_gpu.py --build-dir "$SKY3D_BUILD" \
  --mode prepare --prepare-case 16o-sly5 \
  --prepare-mesh 40 --prepare-spacing 0.6 --maxiter 6000 \
  --output "$SKY3D_RESULTS/16o.json"
python3 CUDA/validate_prepared_tdhf.py --build-dir "$SKY3D_BUILD" \
  --preparation "$SKY3D_RESULTS/16o.json" --case 16o-sly5 \
  --output "$SKY3D_RESULTS/16o-own-gpu.json"
```

The preparation runner generates CPU and GPU static states independently,
requires the original static stopping criterion, checks fresh CPU-field
residuals, Hamiltonian symmetry and occupied subspaces, then compares short
TDHF runs. The own-preparation check also evolves the GPU's own prepared state.
The fixture uses eight CPU threads and is an implementation check, not a
performance benchmark or a production-length response calculation.

Success prints `Single-GPU coverage passed` and produces `"passed": true` in
both reports. Actual wavefunctions, input files and logs are stored in fresh
directories under `~/.cache/sky3d-*`; their paths are recorded in the JSON.
No external checkpoint download is needed. A failed convergence or comparison
preserves its partial report and raw calculation; inspect the recorded
directory and `stdout.log` before using that state.

## 4. Launch TDHF yourself

Use the independently validated GPU static checkpoint from step 3 with the
included [16O TDHF input](../CUDA/examples/16o-tdhf.in). Create a fresh run
directory and copy the checkpoint there:

```bash
cd "$SKY3D_REPO"
export SKY3D_RUN="$(mktemp -d "$HOME/.cache/sky3d-16o-run.XXXXXX")"
cp CUDA/examples/16o-tdhf.in "$SKY3D_RUN/for005"
python3 - <<'PY'
import json, os, shutil
from pathlib import Path
report = json.loads((Path(os.environ['SKY3D_RESULTS']) / '16o.json').read_text())
if not report['passed']:
    raise SystemExit('Static/TDHF validation did not pass')
case = next(row for row in report['cases'] if row['case'] == '16o-sly5')
seed = Path(case['runs']['static-gpu']['directory']) / 'static.tdhf'
shutil.copy2(seed, Path(os.environ['SKY3D_RUN']) / 'initial.tdhf')
print('Calculation directory:', os.environ['SKY3D_RUN'])
PY
python3 CUDA/preflight.py --library "$SKY3D_BUILD/libsky3d_gpu.so" \
  --input "$SKY3D_RUN/for005"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close OMP_PLACES=cores
cd "$SKY3D_RUN"
SKY3D_BACKEND=gpu "$SKY3D_BUILD/gpu/sky3d.gpu" > stdout.log 2>&1
```

Sky3D reads the file named **`for005`** in its working directory. Fragment
filenames are relative to that directory. The example takes 100 steps at
dt=0.1 fm/c, writes energies/multipoles every 10 steps and saves the final
wavefunction as `dynamic.tdhf`. Density snapshots are disabled with `mplot=0`.

Verify completion from the saved checkpoint, since a legacy Fortran `STOP`
can return exit status zero before reaching the requested endpoint:

```bash
python3 - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['SKY3D_REPO']) / 'CUDA'))
from benchmark import checkpoint
state = checkpoint(Path(os.environ['SKY3D_RUN']) / 'dynamic.tdhf')
if state['step'] != 100 or abs(state['time'] - 10.0) > 1e-8:
    raise SystemExit('Requested 10 fm/c endpoint was not reached; inspect stdout.log')
if max(abs(number - 8.0) for number in state['particles']) > 1e-6:
    raise SystemExit('Particle conservation check failed')
print('Reached', state['step'], 'steps at', state['time'], 'fm/c')
print('Neutrons/protons:', state['particles'])
PY
```

Main outputs are `stdout.log`, `energies.res`, `quadrupoles.res`,
`monopoles.res` and `dynamic.tdhf`. To compare speed, prepare another fresh
directory with the same `for005` and `initial.tdhf`, then run
`SKY3D_BACKEND=cpu "$SKY3D_BUILD/cpu/sky3d.cpu"` with identical output and
thread settings. Time complete jobs, repeat them sequentially and compare
their outputs before claiming a speedup.

## 5. Move to your own nucleus

Prepare and qualify that nucleus's static state first. Use a matching force,
mesh and fragment checkpoint in `for005`; select boost amplitude, multipole,
time step and duration for your scientific calculation. For deformed nuclei,
inspect the density's principal axes: native lab M=0 is about z and need not
be intrinsic K=0. Rotation must include spinors. See the
[alignment discussion](SINGLE_GPU_RESULTS.md#static-alignment-and-the-response-frame).

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
| `nvcc` missing | Install a compatible toolkit; set `CUDA_HOME` and `PATH`. |
| Unsupported GPU architecture | Select a toolkit supporting that GPU; check `cuda-build.log` and `--arch`. |
| Missing FFTW/LAPACK/OpenBLAS | Install the development packages above; inspect the variant's `build.log`. |
| Missing `file_digest` or `tomllib` | Use Python 3.11+ in the shell running the scripts. |
| GPU unavailable in WSL | Check the Windows driver and WSL2 setup; run `nvidia-smi` inside WSL. |
| Missing `.so` after moving a build | Rebuild on the destination host; the build contains absolute library paths. |
| Insufficient VRAM | Close competing GPU jobs or use parallel CPU; count all propagated orbitals. |
| No final checkpoint or wrong endpoint | Inspect `stdout.log`; exit status zero alone does not establish completion. |
| Static gate or comparison fails | Preserve the report/raw directory and investigate; do not loosen tolerances automatically. |
| Numerical results differ on another host | Run matched CPU/GPU checks for that host and case before a production calculation. |
