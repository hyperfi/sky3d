# Local research snapshot, 2026-10-08

The `research-workflows` branch records the previously uncommitted response
analysis package and isolated 24Mg and 20Ne workflows. The Fortran physics
source is unchanged. `main` remains at its original commit.

Versioned material includes Python source and tests, exact calculation and
analysis inputs, run/build scripts, manuscript source and its template
provenance, experimental-data provenance, compact CSV/JSON result tables,
and selected 20Ne run provenance and timing records. Saved results describe
earlier calculations; committing them is not a new physical validation.

Compiler objects/modules/executables, Python caches, raw run directories,
restart wave functions, density dumps, generated plots, paper exports,
large HDF5 datasets, and the 24Mg high-resolution spectra export are ignored.
Operational `GO` gate files remain local so a fresh clone does not inherit
authorization to launch a later calculation stage.

No local files were deleted. `LOCAL_ARTIFACTS.json` records sizes and SHA-256
checksums for the newly ignored files from the original untracked inventory.
Existing ignored 24Mg raw runs/build/logs are also retained locally; they are
outside that inventory. Ignored files are not backed up by Git, and their
checksums do not replace the files themselves.

The analysis package was tested in WSL on 2026-10-08: all 10 existing unittest
tests passed, including signed response, baseline subtraction, restart-point
handling, density diagnostics, and CSV/HDF5/plot output. Bash scripts were
syntax checked and Python sources were parsed before committing. These checks
do not rerun the saved production trajectories or validate the manuscript.
