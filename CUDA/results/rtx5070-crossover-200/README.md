# Sky3D TDHF grid sweep on the RTX 5070

The RTX 5070 beat the fastest screened OpenMP configuration in all twelve
200 fm/c pairs, with speedups of 1.51–6.75×. It led at every tested grid
from 24³ to 40³ for oxygen-16, tin-120 and lead-208. The sweep finished on
19 November 2026.

## Workload and timings

The tests used an Intel Core Ultra 7 265K and a GeForce RTX 5070 with 12 GB
VRAM under WSL2. The build used source revision `4267f80`, CUDA 13.0 with
`sm_120`, and gfortran 13.3. The optimized CPU executable used OpenMP and
`-O3 -march=native -ffast-math`. The CUDA numerics used FP64.

Each case reproduces the input recipe from the workstation’s 500 fm/c
sweep, with a 200 fm/c endpoint. Each backend evolves the same locally
prepared CPU checkpoint for 1,000 steps at 0.2 fm/c with a fourth-order
Taylor expansion and 1 fm mesh spacing. Increasing the mesh dimension
changes the box size while keeping the spatial resolution fixed. The
static input caps preparation at 100 iterations with `serr=1e-6`. All
twelve cases reached that cap before static convergence.

The timings cover each complete TDHF job, including initialization, output
every 50 steps and the final checkpoint write. One CPU job and one GPU job
were timed per case. Jobs ran sequentially, with backend order alternating
across successive nuclei and grid sizes. The CPU ran 50-step pilots with
4, 8, 12, 16 and 20 threads per case. The two fastest settings were repeated
in reverse order when their first timings were within 10%. The GPU used
eight host threads and CUDA graphs with the default cache policy.

| Nucleus | Grid | Orbitals | CPU threads | CPU wall time (s) | GPU wall time (s) | CPU/GPU | Endpoint invariants |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Oxygen-16 | 24³ | 16 | 8 | 38.16 | 9.27 | 4.12× | Pass |
| Oxygen-16 | 32³ | 16 | 8 | 121.28 | 21.12 | 5.74× | Pass |
| Oxygen-16 | 36³ | 16 | 8 | 188.59 | 27.92 | 6.75× | Pass |
| Oxygen-16 | 40³ | 16 | 8 | 308.17 | 49.11 | 6.28× | Pass |
| Tin-120 | 24³ | 170 | 16 | 119.59 | 72.98 | 1.64× | Fail on both |
| Tin-120 | 32³ | 170 | 8 | 484.02 | 233.80 | 2.07× | Fail on both |
| Tin-120 | 36³ | 170 | 8 | 916.46 | 230.99 | 3.97× | Fail on both |
| Tin-120 | 40³ | 170 | 8 | 1777.47 | 421.23 | 4.22× | Fail on both |
| Lead-208 | 24³ | 208 | 16 | 134.27 | 88.71 | 1.51× | Fail on both |
| Lead-208 | 32³ | 208 | 8 | 523.89 | 260.05 | 2.01× | Fail on both |
| Lead-208 | 36³ | 208 | 8 | 1125.25 | 287.28 | 3.92× | Fail on both |
| Lead-208 | 40³ | 208 | 8 | 2153.62 | 530.27 | 4.06× | Fail on both |

All twelve pairs reached 200 fm/c and passed the CPU/GPU agreement checks
for endpoint orbitals and the printed energy, quadrupole and monopole
histories. The largest relative orbital L2 difference was 1.11 × 10⁻¹³.
The endpoint conservation check requires both the maximum Gram-matrix error
and the maximum particle-number drift to be at most 10⁻⁶. Oxygen-16 met
both limits. Tin-120 and lead-208 exceeded both limits on each backend.
Their timings characterize throughput on the shared test states. The
benchmark leaves static convergence and physical response convergence open.

The [completed run record](summary.json) preserves the build, pilot
timings, input and checkpoint hashes, numerical comparisons and campaign
status. Directory paths identify files on the original WSL host. Raw
wavefunctions and logs are kept in that host’s cache.

## Reproduce the sweep

Run the following from the repository root under Linux or WSL2, with the
toolchain and libraries from the [GPU build guide](../../../docs/GPU_QUICKSTART.md).
Use the CUDA architecture that matches the installed GPU.

```bash
mkdir -p "$HOME/.cache"
build_dir=$(mktemp -d "$HOME/.cache/sky3d-crossover-build.XXXXXX")
run_dir=$(mktemp -d "$HOME/.cache/sky3d-crossover-200.XXXXXX")
python3 CUDA/build.py --build-dir "$build_dir" --arch sm_120
python3 CUDA/results/rtx5070-crossover-200/run.py "$run_dir" \
  --build "$build_dir" --report "${run_dir}.json"
```

## Comparison between the two systems

For lead-208, the A2000 workstation has a 25-step sweep whose source-file
manifest matches repository revision `4267f80`, the revision used for the
local build. The Xeon w7-3455 used 24 OpenMP threads.
Each workstation value is the median of two jobs per backend with reversed
run order. The Xeon was faster than the A2000 on 24³ and 32³. The A2000 was
1.27× faster on both 36³ and 40³, placing the observed crossover between
32³ and 36³. The RTX 5070 remained ahead of its CPU baseline throughout
the 24³–40³ range, leaving the local crossover unobserved.
[Workstation grid sweep](../a2000/trials/final-grid-sweep/README.md)

| Lead-208 grid | Core Ultra (ms/step) | RTX 5070 (ms/step) | Xeon (ms/step) | RTX A2000 (ms/step) |
| --- | ---: | ---: | ---: | ---: |
| 24³ | 134.27 | 88.71 | 226.48 | 333.98 |
| 32³ | 523.89 | 260.05 | 643.05 | 915.83 |
| 36³ | 1125.25 | 287.28 | 1220.37 | 962.91 |
| 40³ | 2153.62 | 530.27 | 2147.89 | 1687.28 |

The table divides each complete job’s wall time by its step count. The
25-step workstation measurements therefore carry more startup and output
cost per step than the 1,000-step WSL jobs. Checkpoints were prepared
separately on each host. The archived 2,500-step sweep reduces the startup
weight and covers all three nuclei, although it used an earlier GPU build.
The following values use the faster of the two archived A2000 cache
configurations for each row. [Workstation 500 fm/c sweep](../a2000/timing-sweep-500/README.md)

| Nucleus | Grid | Core Ultra (ms/step) | RTX 5070 (ms/step) | Xeon (ms/step) | RTX A2000 (ms/step) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Oxygen-16 | 24³ | 38.16 | 9.27 | 55.23 | 26.10 |
| Oxygen-16 | 32³ | 121.28 | 21.12 | 157.05 | 68.69 |
| Oxygen-16 | 36³ | 188.59 | 27.92 | 235.96 | 106.51 |
| Oxygen-16 | 40³ | 308.17 | 49.11 | 361.75 | 140.89 |
| Tin-120 | 24³ | 119.59 | 72.98 | 146.94 | 238.68 |
| Tin-120 | 32³ | 484.02 | 233.80 | 456.61 | 644.02 |
| Tin-120 | 36³ | 916.46 | 230.99 | 900.60 | 997.98 |
| Tin-120 | 40³ | 1777.47 | 421.23 | 1607.37 | 1331.29 |
| Lead-208 | 24³ | 134.27 | 88.71 | 170.47 | 290.78 |
| Lead-208 | 32³ | 523.89 | 260.05 | 538.99 | 782.92 |
| Lead-208 | 36³ | 1125.25 | 287.28 | 1069.75 | 1227.65 |
| Lead-208 | 40³ | 2153.62 | 530.27 | 1896.21 | 1644.89 |

The RTX 5070 had the lowest time per step in every row. For the heavy
nuclei, the Core Ultra led the Xeon on 24³. On 32³ and 36³ the two CPUs
were within about 6% and traded places. The A2000 trailed the Core Ultra
on 24³–36³ and beat both CPUs on 40³ for tin and lead. It also beat both
CPUs on all four oxygen-16 grids. These rates describe the complete runs on
the recorded systems. A controlled device comparison would require the same source
revision, byte-identical checkpoints, step count and output schedule on
both machines.
