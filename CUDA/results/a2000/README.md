# RTX A2000 CPU and GPU benchmark evidence

The workstation used an NVIDIA RTX A2000 12 GB and an Intel Xeon w7-3455 with 24 physical cores. The main GPU builds used Ubuntu 22.04, CUDA 11.5, `sm_86`, and `/usr/bin/g++-10`. The timing baseline is the optimized `sky3d.cpu` executable. The strict `sky3d.reference` executable was used for selected numerical checks. The final GPU build was tested from an uncommitted worktree based on `ad07395`. Its CUDA source matches `fcbea25`. See [hardware details](hardware.txt) and the build manifests alongside the trial results.

| Case | CPU wall time | GPU wall time | Result |
| --- | ---: | ---: | --- |
| Lead-208, 24³, 6000 fm/c, `ad07395` | 5944.65 s | 12205.09 s | CPU 2.05× |
| Lead-208, 24³, 100 steps, final GPU build | 20.57 s | 33.68 s | CPU 1.64× |
| Lead-208, 40³, 20 resumed steps, final GPU build | 47.79 s | 39.26 s | GPU 1.22× |
| Lead-208, 40³, 3000 fm/c, final GPU build | 31214.86 s | 26615.12 s | GPU 1.17× |

The 24³ 100-step GPU time is the median of two runs. The CPU time is from the matched 100-step input in the Hamiltonian-accumulation trial. The short 40³ timings are the averages of two paired runs in reversed order. Each mesh used 1 fm spacing. The 40³ benchmark state was made by padding the 24³ lead-208 state and evolving it on the CPU for 20 steps to 4 fm/c. These are throughput tests using a shared checkpoint. The static preparation on the 24³ grid did not reach the requested `serr=1e-6`.

## Evidence

- [Lead-208 on 40³ to 3000 fm/c](pb208-40-3000/README.md) contains the final timings, endpoint checks, time series, and the CPU and GPU Fourier response with 1 MeV damping.
- [Heavy nuclei on 24³](heavy-24/summary.json) records the completed lead-208 comparison, the tin-120 CPU trajectory and cancelled GPU trajectory, and the samarium-154 pilot. The lead-208 endpoint metrics show that CPU and GPU agreed and both exceeded the conservation tolerances. The [tin-120 energy comparison](heavy-24/Sn120/shared-energy-comparison.json) shows agreement to the printed precision through the last shared sample at 3522 fm/c.
- [Short trials](trials/) contains matched timings, build manifests, run inputs, observable tables and kernel summaries where available. It includes [rejected warp-per-line transforms](trials/warp-per-line/summary.json), the [CUDA 12.5 screen](trials/grid-screen/summary.json), [24³ timings for the final build](trials/final-grid24/summary.json), and the [matched 40³ restart trials](trials/grid40-restarts/summary.json). The failed mixed-grid trials stopped at the initial collective-flow energy check. The later [rectangular-grid and graph checks](trials/grid40-restarts/summary.json) and [oxygen-16 static smoke check](trials/grid40-restarts/smoke.log) provide the additional validation.

The archived JSON files retain workstation paths as provenance. Wavefunction checkpoints, executables, and full profiler traces are stored on the workstation. The archive includes the inputs, compact logs, extracted kernel tables and numerical comparisons used to report the results.
