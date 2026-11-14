# Lead-208 24³ timestep profile

The RTX A2000 and 24-thread Xeon w7-3455 ran the same 30-step lead-208 input with 208 orbitals and the shared 24³ checkpoint. Both used the build represented by `fcbea25`. The GPU used eight host threads and CUDA graphs. The unprofiled CPU and GPU runs took 7.55 s and 10.68 s. Final wavefunctions agreed to a relative L2 error of 3.24 × 10⁻¹⁵, and the energy and moment tables passed the existing comparisons.

| GPU work | Seconds | Share of kernel time |
| --- | ---: | ---: |
| Y and Z Fourier derivative kernels | 4.432 | 49.4% |
| cuFFT and spectral scaling | 2.265 | 25.2% |
| `h_axis` Hamiltonian kernels | 1.579 | 17.6% |
| Other kernels | 0.699 | 7.8% |
| Total kernel time | 8.974 | 100% |

The Nsight Systems run took 12.67 s including profiling overhead. Reported host-to-device and device-to-host copy time totaled 0.024 s. Device-to-device copies took 0.089 s. The device kernels alone took longer than the unprofiled CPU job.

A separate CPU build with wall timers took 7.57 s overall and 5.907 s in the timestep loop. The predictor and corrector propagation and density loops together took 3.547 s. The CPU processes orbitals in parallel with OpenMP and operates on one orbital at a time within each worker. The GPU executes each derivative and Hamiltonian pass across the full batch. On this grid, a single orbital is 432 KiB and the full batch is 87.75 MiB. That layout gives the CPU an opportunity to reuse data in cache. Cache-hit rates, GPU memory throughput, and FP64 instruction utilization were unavailable because the workstation blocks hardware counters for this account.

The trace points to orbital derivatives as the first target for further work. Both backends use double precision. The profile does not resolve how much of the derivative time is spent on arithmetic, warp shuffles, or memory traffic.
