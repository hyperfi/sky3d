# Single-GPU completion plan

Work continues on branch `gpu`. Multi-GPU and MPI offload come last and are
outside this plan. Builds, numerical runs and analysis use WSL. Keep raw
states, densities, compiler products and logs in the local cache; commit
source, scripts, documentation and small evidence summaries only.

## Starting point

Baseline: `0bee4c6`. The FP64 CUDA C++ backend already accelerates dynamic
Hamiltonian application, spectral derivatives, predictor/corrector propagation
and density accumulation. Field construction, Coulomb, diagnostics and static
preparation still use CPU code. Existing tests cover rectangular padding,
center-of-mass resets, a restart and CUDA memory checking.

The measured 40³/0.6 fm Sly5 20Ne workload takes 43.45 s on GPU versus 62.60 s
for the best measured CPU configuration: 1.441× for 200 steps at dt=0.1 fm/c.
This was the starting local result, not a target or a production-length guarantee.
See [the baseline report](FINE_GRID_GPU_RESULTS.md). CPU fields and diagnostics
account for about half the instrumented GPU job; they guide the next work.

## 1. Strengthen the reference and validation matrix

Start with the settled 40³ 20Ne state, preserving its hash and fresh-field
convergence evidence. Run CPU and GPU with identical settings for each case:

- Extend evolution from 20 to 100 fm/c; check the actual saved endpoint,
  finite observables, particle conservation and occupied-state orthogonality.
- Compare boosted and unboosted evolution at 10 fm/c against the strict
  one-thread CPU reference and optimized parallel CPU build.
- Exercise dt=0.1 and 0.05 fm/c and Taylor orders 4 and 6 at the same physical
  endpoint. Separate CPU/GPU equivalence at fixed settings from numerical
  differences caused by changing integration settings.
- Audit the time level of printed energies against fields recomputed from the
  saved density. `tinfo` precedes the final `skyrme` refresh, while integrated
  Coulomb energy uses `wcoul`; measure this effect before interpreting printed
  drift as physical energy nonconservation. Resolve diagnostic timing separately
  from propagation changes, with matched CPU/GPU checks.
- Extend restart coverage to multiple segments and different diagnostic
  intervals. Include center-of-mass resets and existing rectangular controls.
- Add a second light nucleus, then a second force and nontrivial occupations.
  Prepare and validate their static states independently before dynamic tests.

**Gate:** comparisons retain the existing documented CPU/GPU tolerances. Set
physical conservation and integration-accuracy limits explicitly, using CPU
controls; record failures without automatically relaxing tolerances. A finite
run or CPU/GPU agreement alone does not establish physical convergence.
The initial 20Ne block is only part of this gate. A 100 fm/c trajectory remains
shorter than a production response calculation.

**Deliverables:** reproducible validation runner, small JSON reports, explicit
coverage/limitations and a commit. Raw output stays in WSL's cache.

## 2. Move field construction onto the GPU

Port the measured field work in increments: local Skyrme terms, density
derivatives/spin/time-odd terms, then Coulomb. Keep densities and fields resident
through the predictor/midpoint/corrector sequence; copy them to CPU when output
or remaining CPU functionality requires them.

Use the same force coefficients, discrete derivative conventions, Nyquist
treatment and expanded effective-mass operator as the CPU code. Do not change
the physics operator to obtain a faster result. Validate intermediate fields
and Hψ, rather than relying only on final energies. Preserve external-field
timing and center-of-mass reset semantics.

For isolated Coulomb, preserve the doubled box and existing Green-function
normalization/origin value; uploading the CPU-generated kernel once avoids
changing initialization. Cover periodic Coulomb separately before advertising
that mode. Include the extra buffers and measured cuFFT workspace in the VRAM
guard. Unsupported paths must fail clearly or take an explicit CPU fallback.

**Gate:** initial and intermediate fields, Hψ, final wavefunctions, densities
and observables agree with CPU across stage 1 and existing controls. CUDA
memory checking passes. Run fresh complete-job benchmarks after each useful
increment; retain an optimization only when its cost is justified by results.

**Deliverables:** field kernels, Fortran integration, updated memory estimate,
validation/profiling evidence and separate reviewable commits.

## 3. Reduce diagnostics and transfers

Profile the remaining work after the field port. Accelerate occupied-state
norm/energy calculations, single-particle properties, moments and reductions
where measurements justify it. Consider CPU parallelization when it is the
simpler effective option, and include that improvement in the CPU benchmark.
Keep wavefunctions resident until actual CPU access or file output needs them.

**Gate:** preserve output definitions and frequencies, file compatibility,
restart behavior and the explicit numerical tolerances. Test diagnostic
intervals including disabled output, and rerun conservation/restart checks.
Benchmark both backends with matched diagnostic and checkpoint settings.

**Deliverables:** diagnostic acceleration and transfer changes, comparisons,
updated stage profile and a commit.

## 4. Accelerate static preparation

Reuse the validated GPU Hψ/density/field path in static iteration. Preserve
the read-only basis used for Hamiltonian matrix construction, relaxation and
orthogonalization order, per-species occupations and convergence bookkeeping.
Start with the small occupied-space diagonalization on CPU; move it only if
profiling supports doing so. Pairing/occupation updates require their own tests.

**Gate:** compare density, energy, occupied subspace and freshly recomputed
residual/variance with the CPU result. Individual orbitals may rotate within
degenerate subspaces; compare the subspace appropriately. Require the original
explicit convergence threshold, report iteration exhaustion clearly, and
measure total time to a validated state. Do not infer convergence from the
saved fluctuation alone or relax a tolerance to make the GPU run pass.

**Deliverables:** static GPU path with explicit supported modes, convergence
tests, matched time-to-solution evidence and a commit.

## 5. Final single-GPU validation and clone readiness

Run the full matrix on the final implementation, plus a longer response case
with matched CPU input and native multipole/orientation conventions. Assess
time-step sensitivity, physical conservation and extracted observables before
calling it suitable for production. Validate any added nuclei/forces separately.

Rebuild cleanly, run memory checks and test fresh-start/restart workflows.
Benchmark at least three sequential, interleaved repetitions against the best
measured current parallel CPU configuration. Include startup, diagnostics and
I/O identically; use uninstrumented builds for speed claims. Preserve input,
state, executable and source hashes.

Update the preflight/test launcher for all new allocations and propagated
states, including empty states. Keep conservative free-VRAM warnings and
refuse oversubscription. Distinguish a memory-fit estimate from a speedup
prediction: benchmark the requested case when speed is unknown. Give explicit
CPU guidance when available measurements favor parallel CPU.

Provide clone/build/run instructions, supported/unsupported modes, small test
inputs and result summaries. Keep large generated files out of Git. Prepare
the final branch for review and push when the user is satisfied, as previously
requested. Larger-machine validation remains future work until measured there.

**Gate:** all declared supported modes pass the final checks; limitations and
actual speedups are documented. A source build alone does not satisfy this gate.

## Execution status

- Baseline hybrid implementation and fine-grid checkpoint: complete.
- Stage 1: 20Ne fixed-setting, rectangular/reset, pulse, output-schedule and
  segmented-restart checks pass on the final backend. 16O/SLy5 and 20Ne/SLy4
  with VDI fractional occupations pass fixed-iteration implementation controls.
  The final 17-job integration/unboosted/100 fm/c matrix passes. Independent
  converged-state and own-GPU-checkpoint TDHF checks pass for 16O/SLy5 and
  paired 20Ne/SLy4; refinement/timestep failures remain explicitly recorded.
- Stage 2: complete. Skyrme, real-field derivatives and isolated/periodic
  Coulomb are on GPU; periodic/disabled-Coulomb controls pass. The field-only
  milestone measured 2.0603x against CPU8; see [GPU_FIELD_PORT.md](GPU_FIELD_PORT.md).
- Stage 3: complete. Resident predictor/corrector densities, M=0 reductions,
  single-particle properties and field graphs pass component/pulse/restart
  controls and CUDA memcheck. Other M projections retain native CPU diagnostics.
  The final fine-grid 200-step benchmark measures 3.1879x against the best
  measured CPU4/8/20 configuration (three interleaved complete jobs each).
- Stage 4: complete. GPU damped gradients, static Hamiltonians, densities
  and fields pass 120-iteration controls. Ordered orthogonalization, pairing,
  basis overlaps and small diagonalization retain CPU execution. Fresh-field
  convergence, subspace and time-to-solution checks pass. The independent
  static speedups are 1.6243x for 16O and 1.7948x for paired 20Ne, one pair
  each. Static fallbacks pass with small diagonalization both on and off.
- Stage 5: backend comparisons, profiling, static/dynamic memory checking,
  refusal checks and 19 unit tests are complete. A clean local clone of
  `c939eb5` rebuilds all variants and passes 10-step standard/rectangular/reset
  strict-CPU comparisons. All 337 protected local artifacts retain their
  original hashes; `main` and `origin/main` are unchanged. Implementation and
  reproducibility work is complete. The 6000 fm/c K=0 replay matches CPU and the existing local spectrum,
  but its archived coarse CPU and GPU trajectories both fail the unchanged
  physical Gram gate (6.4175e-5 versus 1e-6). This remains a failed scientific
  qualification gate; refined long-response convergence is future physics
  validation, not an accepted production result or a relaxed GPU tolerance.
- Multi-GPU: deferred until later, as requested.

See [SINGLE_GPU_RESULTS.md](SINGLE_GPU_RESULTS.md) and the linked machine-readable
reports. Update the status only when the corresponding gates finish.
