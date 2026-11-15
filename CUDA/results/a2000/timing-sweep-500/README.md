# Sky3D CPU and GPU timing across nuclei and grids

The sweep uses oxygen-16, tin-120 and lead-208 on 24³, 32³, 36³ and 40³ grids at 1 fm spacing. This changes the box size while holding the spatial resolution fixed. The force is SLy5, with the existing VDI pairing setup for tin-120. The workstation has an RTX A2000 12 GB and a Xeon w7-3455. CPU jobs use the optimized build with 24 OpenMP threads. GPU jobs use eight host threads and CUDA graphs. Jobs run sequentially.

## Timing definitions

- **Static** measures the iteration loop directly, with a cap of 100 iterations and a convergence target of `serr=1e-6`. The preparation wall time, achieved iteration count and convergence status are in the timing data. These runs provide a fixed-budget timing comparison.
- **Dynamics** measures wall time for 2,500 steps at 0.2 fm/c. All three paths for each grid and nucleus start from the same CPU static checkpoint. GPU runs compare the full predictor–corrector derivative cache enabled and disabled. The experimental setting allows caching at 24³, 32³, 36³ and 40³ within the existing memory budget. The reported GPU memory requirement shows whether caching was active.
- **Total** adds the full static job wall time for each backend to its matched dynamics time. Both GPU totals use the same GPU static measurement because the cache only affects propagation. These are phase sums with shared dynamics inputs.

The timing campaign uses one measurement per series and grid. The curve colors identify nuclei and the line styles distinguish CPU, GPU cache off, and GPU cache on. The archive includes exact inputs, benchmark scripts, build details, numerical comparisons and the three figures as PNG and PDF. Checkpoints are identified by SHA-256 in the summary.

The 36³ extension reused the original executables and input-generation rules. The other three grids retain their original measurements in the updated figures. The initial 45-run summary is preserved in summary-original.json. At 36³ the Fourier derivatives use the generic kernel path, while the 40³ runs use the specialized 40-point Y/Z kernels.
