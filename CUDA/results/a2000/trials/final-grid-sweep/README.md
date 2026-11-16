# Lead-208 CPU and GPU grid timings

The benchmark advances 25 TDHF steps at 0.2 fm/c with 1 fm grid spacing. Each grid uses its CPU-prepared checkpoint from the [500 fm/c sweep](../../timing-sweep-500/README.md). The optimized Xeon w7-3455 CPU build uses 24 threads. The RTX A2000 GPU build uses eight host threads and CUDA graphs. Each reported time is the median of two complete dynamics jobs, with CPU and GPU run order reversed in the second pair. Times include process startup and checkpoint output. The public build’s cache policy was used, with the derivative cache activated at 24³ and disabled at the other grids.

| Grid | CPU | GPU | Faster backend |
| --- | ---: | ---: | --- |
| 24³ | 5.66 s | 8.35 s | CPU 1.47× |
| 32³ | 16.08 s | 22.90 s | CPU 1.42× |
| 36³ | 30.51 s | 24.07 s | GPU 1.27× |
| 40³ | 53.70 s | 42.18 s | GPU 1.27× |

The [X-axis experiment](x-axis-trial.json) held the rest of the GPU build fixed. Over two reversed-order 25-step pairs per grid, the fused X derivative reduced median wall time from 25.24 s to 24.03 s at 36³ and from 44.62 s to 43.75 s at 40³. The final implementation uses the existing Fourier factorization for contiguous X lines and applies spectral factors within the same kernel.

All eight timed GPU runs met the energy, quadrupole, monopole, and wavefunction tolerances against the corresponding CPU outputs. The largest relative wavefunction L2 error was 2.78 × 10⁻¹⁵. The directly padded rectangular initialization exceeded the initial collective-flow energy tolerance by showing a 6.04 × 10⁻⁵ MeV difference. The 36 × 32 × 24 and 40 × 32 × 24 graphs-off retests from shared, six-step checkpoints passed the original tolerances. All 25 CUDA Python tests passed. See [individual timings and checks](summary.json) and the [timings CSV](timings.csv).
