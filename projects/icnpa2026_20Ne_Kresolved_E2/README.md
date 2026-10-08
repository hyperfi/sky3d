# 20Ne K-resolved isoscalar E2 response

This project is isolated from `projects/mg24_mqc_snp2026` and reuses its
validated Sky3D v1.2 executable and response-analysis package without changing
either one. All compilation, execution, and analysis commands are run in WSL.

## Reproduce

From the repository root in WSL:

```bash
cd /mnt/d/Coding/sky3d
projects/icnpa2026_20Ne_Kresolved_E2/scripts/run_all_wsl.sh
```

On a 20-logical-CPU host the five independent production trajectories can be
launched together with four OpenMP threads each after the static state has been
aligned:

```bash
OMP_THREADS_PER_CASE=4 \
  projects/icnpa2026_20Ne_Kresolved_E2/scripts/run_production_parallel_wsl.sh
```

The runner refuses to overwrite non-empty calculation directories. The static
state is completed and inspected before any TDHF run is launched. The common
production protocol is 24^3 points at 1 fm spacing, SLy5 without pairing,
`dt=0.2 fm/c`, and `T=6000 fm/c`. The K=0 half-amplitude trajectory is the
linearity check.

The production executable is reused from
`projects/mg24_mqc_snp2026/build/sky3d.omp`; its checksum and the repository
commit are recorded in `logs/provenance.txt` by the runner.
