# Lead-208 Fourier kernel comparisons

These runs use the 36³ and 40³ lead-208 CPU-prepared states from the [500 fm/c sweep](../../timing-sweep-500/README.md), at 1 fm grid spacing and a timestep of 0.2 fm/c. The same RTX A2000 and 24-thread Xeon were used. Each GPU median comes from two 50-step runs in reversed order.

| Comparison | Generic Y/Z Fourier path | Specialized Y/Z Fourier path | Wall time saved |
| --- | ---: | ---: | ---: |
| 36³, experimental derivative cache on | 65.99 s | 51.79 s | 21.5% |
| 40³, experimental derivative cache on | 105.01 s | 89.07 s | 15.2% |
| 36³, local branch cache settings | 69.48 s | 54.24 s | 21.9% |

The 36³ experiment adds a fused 36-point derivative based on a 4 × 9 Fourier factorization. It reads and writes the strided Y/Z lines directly and avoids separate packing and unpacking. The 40³ experiment disables the existing 40-point kernel under otherwise identical settings. See [ablation timings and profile totals](ablation.json) and the per-kernel CSVs in each profile folder.

The final 36-point patch uses the local branch’s cache policy. The matched 36³ CPU run took 64.86 s, giving 1.20× GPU throughput on this 50-step comparison. The new GPU path agreed with the CPU to about 2.25 × 10⁻¹⁵ relative wavefunction error. A rectangular 40 × 36 × 40 case with CUDA graphs disabled and a 40³ regression passed. See [final timings and comparisons](final-36.json).

The 500 fm/c sweep uses the earlier build. The 36³ result here is a short dynamics comparison.

The 40-point trial also separated first-derivative launches from launches that compute both derivatives. This reduced the first-derivative kernel's register count from 72 to 48 per thread. Median wall time over two 40-step runs was 72.20 s before and 72.35 s with the split. The existing 40-point kernel stays in use. See [split-kernel timings](split-40.json).
