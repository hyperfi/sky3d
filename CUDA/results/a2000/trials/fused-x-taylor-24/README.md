# Fused X derivative and Taylor update

The RTX A2000 ran lead-208 on a 24³ grid from the same checkpoint, with eight host threads and CUDA graphs. Each comparison used two 100-step timing pairs, with the second pair reversing the run order.

| Experiment | Baseline median | Candidate median | Wall time saved |
| --- | ---: | ---: | ---: |
| Fused X derivative | 30.47 s | 30.02 s | 1.5% |
| Taylor update fused with final Z derivative | 31.11 s | 30.57 s | 1.7% |
| Combined implementation | 31.17 s | 30.03 s | 3.7% |

The combined implementation reuses the existing warp Fourier routine for contiguous 24-point X lines. On 24³, the last Z derivative also writes the next Taylor term and updates the propagated wavefunction. Both timings for the combined build beat their paired baselines. The wavefunctions agreed with the strict CPU reference to about 1.1 × 10⁻¹⁴ relative error.

The combined build passed the CUDA graphs-off, low-memory cache fallback, order-one propagation, 24 × 32 × 24 and 40³ trajectory checks. All 25 CUDA Python tests passed. See the [combined timings](summary.json), [X-axis trial](x-axis.json) and [Taylor update trial](taylor-update.json).
