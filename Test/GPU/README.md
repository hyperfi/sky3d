# Oxygen-16 GPU demo

Prepare the SLy5 ground state, then evolve it for 100 steps after a
quadrupole boost. Both inputs use a 40³ grid with 0.6 fm spacing.

After building with the [GPU build guide](../../docs/GPU_QUICKSTART.md), run
this block from the repository root in Linux or WSL2 Bash. It uses
`CUDA/build/gpu/sky3d.gpu` and creates a fresh `Test/GPU/run.*` directory.

```bash
export SKY3D_RUN="$(mktemp -d "$PWD/Test/GPU/run.XXXXXX")"
printf 'Run directory: %s\n' "$SKY3D_RUN"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export OMP_PROC_BIND=close OMP_PLACES=cores

(
  set -e
  sky3d="$PWD/CUDA/build/gpu/sky3d.gpu"
  cd "$SKY3D_RUN"

  cp ../for005.static for005
  SKY3D_BACKEND=gpu "$sky3d" > static.log 2>&1
  grep 'Static convergence criterion met' static.log
  test -s initial.tdhf

  cp ../for005.tdhf for005
  SKY3D_BACKEND=gpu "$sky3d" > dynamic.log 2>&1
  grep -E 'Wrote restart file at end of +iter= *100$' dynamic.log
  test -s dynamic.tdhf
  tail -n 1 energies.res
)
```

Sky3D reads `for005` from its working directory. The static run writes
`initial.tdhf` and reports convergence at `serr=1e-6` MeV. The dynamic run
reads that checkpoint, writes `dynamic.tdhf` at step 100, and produces
`energies.res`, `quadrupoles.res`, and `monopoles.res`. The last row of
`energies.res` starts with `10.00  8.000  8.000` (time in fm/c, neutrons,
protons). Check `static.log` or `dynamic.log` in the printed run directory
if a completion message is missing.

## Optional CPU reference comparison

After the GPU run, use the same terminal at the repository root. The CPU
reference uses the same static checkpoint and TDHF input. The comparison needs NumPy.
It reads particle counts from the saved wavefunctions and uses the CUDA
validator's existing checkpoint and observable tolerances.

```bash
(
  set -e
  repo="$PWD"
  reference="$(mktemp -d "$SKY3D_RUN/reference.XXXXXX")"
  cp Test/GPU/for005.tdhf "$reference/for005"
  cp "$SKY3D_RUN/initial.tdhf" "$reference/initial.tdhf"
  cd "$reference"
  SKY3D_BACKEND=cpu "$repo/CUDA/build/reference/sky3d.reference" > dynamic.log 2>&1

  PYTHONPATH="$repo/CUDA${PYTHONPATH:+:$PYTHONPATH}" python3 - "$reference" <<'PY'
import os
import sys
from pathlib import Path

import numpy as np

from benchmark import check_observable, checkpoint, difference

gpu = Path(os.environ["SKY3D_RUN"])
reference = Path(sys.argv[1])
gpu_state = checkpoint(gpu / "dynamic.tdhf")
cpu_state = checkpoint(reference / "dynamic.tdhf")
for state in (gpu_state, cpu_state):
    assert state["step"] == 100 and abs(state["time"] - 10.0) < 1e-8
    assert max(abs(count - 8.0) for count in state["particles"]) <= 1e-6

wave = difference(gpu_state["psi"], cpu_state["psi"])
assert wave["relative_l2"] < 1e-9 and wave["max_abs"] < 1e-9, wave
for name in ("energies.res", "quadrupoles.res", "monopoles.res"):
    actual = np.loadtxt(gpu / name, ndmin=2)
    expected = np.loadtxt(reference / name, ndmin=2)
    check_observable(name, actual, expected, initial_flow_atol=1e-5)

print("GPU and CPU reference agree at 100 steps (10 fm/c).")
PY
)
```

For independent CPU/GPU ground-state checks and full-field comparisons,
see the [release validation commands](../../docs/GPU_QUICKSTART.md#4-optional-release-validation).
