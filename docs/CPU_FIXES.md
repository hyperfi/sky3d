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

These are control and accumulation fixes, not a change to the Skyrme force,
spectral derivative rules or propagator. Existing production files are kept
unchanged. The historical GPU audit describes the original source and should
be read together with this note.
