# CPU control fixes before the GPU pilot

The `gpu` branch repairs three issues from the 2026-10-08 audit:

- Clear all five densities after the optional center-of-mass phase reset,
  before reaccumulating. A two-step 20Ne regression now retains 10 neutrons
  and 10 protons with `mrescm=1`, instead of doubling them.
- Use a guarded `output_due` predicate for dynamic/static restart and print
  intervals. Zero disables periodic output without evaluating a zero-divisor
  MOD. Fortran logical operators do not guarantee short-circuit evaluation.
- Expand input fragment filenames from 64 to 1024 characters. This changes
  namelist storage only, not the binary wave-function format.

WSL regression checks cover the normal case, center-of-mass correction, zero
restart/plot/print intervals, a fragment path longer than 64 characters,
restart output every step, and a three-iteration static run with zero print
and restart intervals. Initial state output remains part of initialization;
`mrest=0` disables subsequent periodic restart writes. Evidence is saved in
`tools/gpu_audit/evidence/controls-fixed.json`; the original failing evidence
is preserved in `controls.json`.

Fresh-state preparation also exposed a static diagonalization race. `grstep`
read other orbitals to construct `hmatr` while the OpenMP loop updated those
orbitals in place. The matrix is now assembled in a separate read-only pass
before the gradient loop, using the same pre-update basis for every column.
This adds a Hamiltonian application for iterations that diagonalize, without
copying the full basis or defeating `tlarge`'s memory-saving transformation.
The dynamic propagation path is unchanged by this fix. A 120-iteration static
check compares all seven density/potential outputs across one thread, repeated
eight-thread reference runs and the optimized CPU build. Maximum field
differences are below `4e-13`; see `CUDA/results/static.json`.

Static convergence fluctuations are also updated every iteration, independently
of `mprint`. Otherwise disabling prints leaves the initially zero fluctuation
unchanged and falsely stops at iteration two. The zero-output control test now
checks that a three-iteration run actually reaches iteration three. Computing
the existing fluctuation measure every iteration adds static work; the TDHF
timing path and the definition of the convergence measure are unchanged.

The static race fix does **not** establish that the supplied 24³ 20Ne input
meets its requested `serr=1e-6`. Both the earlier build and the repaired build
reached 3000 iterations at a weighted fluctuation of about `7.56e-5`. The
original production input is retained. The benchmark preparation script
rejects unmet convergence criteria; an explicit `--static-serr 1e-4` generates
a performance-test state and records that tolerance. It does not certify a
production ground state or silently relax a physics criterion.

The subsequent [static convergence study](STATIC_CONVERGENCE.md) attributes
this state's plateau to the expanded effective-mass term on the coarse mesh.
A settled 40³/0.6 fm candidate meets the original criterion in saved and
fresh fields. Static exit now reports success or an iteration-limit warning
and both fluctuation measures. Four success/exhaustion checks with printing
enabled/disabled preserve the pre-change wavefunctions; see
`CUDA/results/static-status.json`. The stopping definition is unchanged.

These are control and accumulation fixes, not a change to the Skyrme force,
spectral derivative rules or propagator. Existing production files are kept
unchanged. The historical GPU audit describes the original source and should
be read together with this note.
