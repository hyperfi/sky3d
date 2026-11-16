# Paired-spin Hamiltonian update

For a given orbital and grid point, one CUDA thread updates both spin components. It reuses their wavefunction values, first derivatives, and mean fields. The X-axis pass also initializes the local Hamiltonian, which removes a separate full-grid kernel. The tested implementation selects this path on 36³ and 40³ grids. The timing comparison uses the existing cache policy, which leaves the derivative cache disabled on both grids.

| Lead-208 grid | CPU, 24 threads | Previous GPU | Updated GPU | GPU wall time saved |
| --- | ---: | ---: | ---: | ---: |
| 36³ | 33.23 s | 28.97 s | 26.17 s | 9.7% |
| 40³ | 57.44 s | 49.08 s | 44.58 s | 9.2% |

Each timing advances 25 TDHF steps from the same grid-specific CPU checkpoint used in the 500 fm/c sweep. GPU times are medians from two pairs in reversed order on an RTX A2000 with eight host threads. Each CPU time is a single matched run. The [final comparisons](summary.json) include wavefunction agreement against the matched CPU outputs, CUDA graphs disabled, and a rectangular-grid regression. The [static smoke check](static-smoke.json) compares densities after 10 oxygen-16 iterations at 40³. All 25 CUDA Python tests passed.

In the [40³ baseline trace](profile-40-baseline.csv), the Hamiltonian axis and local kernels used 7.64 s over 20 steps. The [paired-spin trace](profile-40-candidate.csv) measured 4.54 s for the combined kernels. The [prototype timings](prototype.json) record the separate effect of pairing the spin calculations before adding local-term fusion. These are short-run results. The archived 500 fm/c sweep used an earlier build.
