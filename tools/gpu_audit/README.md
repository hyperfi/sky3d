# Local GPU feasibility audit

These tools build/run in WSL and retain generated executables and raw run files
under `/home/abhishek/.cache/`. They do not alter `Code/` or production runs.
Only small JSON/CSV evidence files are versioned here.

```bash
cd /mnt/d/Coding/sky3d
PYTHONDONTWRITEBYTECODE=1 python3 tools/gpu_audit/profile_cpu.py --steps 100 --repetitions 3
# Wait for the CPU audit to finish before running the GPU probe.
bash tools/gpu_audit/run_gpu_probe.sh
```

`profile_cpu.py` requires the existing local 20Ne aligned restart state and
gfortran, FFTW, LAPACK and OpenBLAS. It copies and builds the Fortran source,
times selected routines in a serial diagnostic build, and measures separate
complete uninstrumented OpenMP runs. Timer totals include nested callees and
must not be added. Instrumented serial shares do not establish threaded shares.
The copied original build flags retain `-ffast-math`; this is the existing
performance baseline, not a new strict floating-point reference.

`run_gpu_probe.sh` uses CUDA 13.0 explicitly and compiles for `sm_120`. It runs
on CUDA device 0. The standalone C++/CUDA probe computes all three first
spectral derivatives in FP64, preserving the first-derivative Nyquist zero
and normalization. Its CPU side mirrors Sky3D's FFTW plan layouts and execution
granularity, with OpenMP over states. It is not the original Fortran routine.
The GPU side batches all states and packs/unpacks each axis. It compares
resident-data timing with a full input upload and three output downloads.
Timings are medians of three trials after warmup; planning/allocation are
excluded. This is a kernel feasibility measurement, not a TDHF speedup or
physical validation. CPU and GPU trials are short and sequential rather than
a full randomized thermal/performance study.

The audit also reproduced optional control bugs with `check_controls.py`,
using the copied CPU build named in `evidence/cpu.json`:

```bash
python3 tools/gpu_audit/check_controls.py --exe /absolute/path/to/copied/cpu-build/sky3d.audit
```

Use a short fragment filename or symlink: this Sky3D version stores fragment
filenames in a 64-character field. The runner uses `../initial.tdhf` to avoid
silent path truncation.
