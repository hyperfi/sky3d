# Occupied-orbital overlap drift in the 20Ne response

Reducing the timestep does not remove the overlap error of the archived
24³, 1 fm SLy5 state. At 200 fm/c, the unboosted and quadrupole-boosted
trajectories have a maximum occupied Gram-matrix error of about 9.3e-5
for timesteps of 0.2, 0.1 and 0.05 fm/c and Taylor orders 4 and 6. The
acceptance limit remains 1e-6. These controls support a spatial
discretization explanation for the dominant error in this case.

The relevant observable is the overlap of orbitals within each species:

$$G_{ij}=\Delta x\Delta y\Delta z\sum_{\mathbf r,\sigma}
\psi_i^*(\mathbf r,\sigma)\psi_j(\mathbf r,\sigma).$$

Diagonal errors measure normalization; off-diagonal entries measure loss
of mutual orthogonality. The study records them separately. Particle
conservation alone cannot detect a growing off-diagonal overlap.

## Measured recovery on the finer state

The settled, z-aligned 40³/0.6 fm state passes the endpoint overlap and
particle gates at 200 fm/c for both 0.1 and 0.05 fm/c timesteps. The largest
Gram error is 7.03e-9, about 13,000 times smaller than the coarse-state
error. Halving the timestep mainly reduces the remaining particle drift;
the finer-state overlap error is already below the unchanged 1e-6 limit.

| Mesh / spacing | Timestep, fm/c | Taylor order | Largest endpoint Gram error | Invariant gates |
| --- | ---: | ---: | ---: | --- |
| 24³ / 1 fm | 0.2 | 4 | 9.32e-5 | Fail |
| 24³ / 1 fm | 0.2 | 6 | 9.32e-5 | Fail |
| 24³ / 1 fm | 0.1 | 4 | 9.32e-5 | Fail |
| 24³ / 1 fm | 0.1 | 6 | 9.32e-5 | Fail |
| 24³ / 1 fm | 0.05 | 4 | 9.32e-5 | Fail |
| 24³ / 1 fm | 0.05 | 6 | 9.32e-5 | Fail |
| 40³ / 0.6 fm | 0.1 | 4 | 7.03e-9 | Pass |
| 40³ / 0.6 fm | 0.05 | 4 | 7.03e-9 | Pass |

Each row takes the maximum over boosted and unboosted runs and both species.
These are endpoint diagnostics, not maxima over the full trajectory.
The largest finer-state particle drift is 3.33e-9 at dt=0.1 and 1.04e-10
at dt=0.05. Matched-baseline quadrupole histories differ by 3.14e-5 in
relative L2 (0.00314%) between the two timesteps, with a maximum difference
of 2.93e-6 fm². This measures sensitivity over 200 fm/c; neither timestep
is declared a converged long-response reference.

A separate 20 fm/c CPU/GPU comparison starts from the same finer,
z-aligned checkpoint. Both boosted and unboosted pairs pass, with
wavefunction relative L2 differences below 7.7e-15 and occupied Gram
differences below 4.0e-15. The actual GPU memory query requires 0.404 GiB.
The short comparison overlaps part of the longer scan; its timings are
diagnostic costs, as recorded in
[execution-protocol.json](../CUDA/results/orthogonality/execution-protocol.json).

The tested starting configuration for further 20Ne response work is the
settled 40³/0.6 fm state, dt=0.1 fm/c and Taylor order 4, with dt=0.05
as a refinement control. This preserves the implemented Hamiltonian and
propagator. It does not qualify a 6000 fm/c response spectrum.

## Why timestep refinement can leave an error floor

For orbitals evolving under the same instantaneous discrete Hamiltonian,

$$\frac{dG_{ij}}{dt}=\frac{i}{\hbar c}
\langle\psi_i|(H^\dagger-H)|\psi_j\rangle,$$

where time is expressed in fm/c and the Hamiltonian in MeV. A Hermitian
Hamiltonian preserves the Gram matrix under exact propagation. A finite
Taylor expansion adds a separate approximation error. The original
[Sky3D paper, section 2.5.2](https://arxiv.org/pdf/1310.5946), describes
the predictor/corrector expansion and the use of norm and energy
conservation to assess timestep and Taylor-order choices.

The [fresh-field static study](STATIC_CONVERGENCE.md) independently
reconstructs the effective-mass operator in its implemented expanded form,

$$-\sum_a[(D_aB)(D_a\psi)+B D_a^2\psi].$$

The discrete Fourier derivatives do not satisfy the exact continuum
product rule for these finite-grid products. On the archived coarse state,
the occupied-space anti-Hermitian component is about 4.3e-4 MeV. The
effective-mass term accounts for it; the remaining projected terms agree
with Hermitian symmetry to roundoff. Changing timestep improves the
approximation to this discrete evolution without eliminating its
non-Hermitian component.

These are occupied-space measurements, not a bound on the full
Hamiltonian. The new controls measure endpoint error at 200 fm/c; they
do not measure its maximum over every intermediate step.

## Controls and retained evidence

All runs start from a byte-identical copied checkpoint. The force, box,
boost, output interval and numerical implementation are fixed within each
scan. The coarse scan uses the archived z-aligned state. Each timestep/order
combination has both an unboosted control and an isoscalar quadrupole
boost of amplitude 5e-5 with L=2, M=0. Output is sampled every 2 fm/c.
The response subtracts the matched unboosted history and its initial
offset; no rescaling or fitted correction is applied.

The numerical source and executable/library hashes are checked before
execution. Failed conservation gates remain recorded failures. A zero
process exit code cannot establish completion: both the saved step/time
and printed history must reach the requested physical endpoint.

The new runner is [study_orthogonality.py](../CUDA/study_orthogonality.py).
Small JSON reports and comparison figures are retained under
[CUDA/results/orthogonality](../CUDA/results/orthogonality). Large
wavefunctions, complete inputs, Gram matrices and logs stay in the WSL
cache directories identified by each report. Run timings are diagnostic
costs, not repeated performance benchmarks.

## Reproduce the timestep and order scan

Use Linux/WSL, a locally compiled GPU branch and a qualified 20Ne checkpoint:

```bash
python3 CUDA/study_orthogonality.py \
  --build-dir /path/to/local/build \
  --state /path/to/20ne_sly5_zaxis.tdhf \
  --duration 200 --sample-interval 2 \
  --dt 0.2 0.1 0.05 --order 4 6 --backends gpu \
  --output /path/to/evidence/coarse-200.json
```

Use `--backends cpu gpu` for paired implementation comparisons. The
runner requires SLy5 20Ne with ten unit-occupied orbitals per species.
It retains invariant failures in the report and continues the scan;
inspect `completed` and `invariant_passed` for every case. A completed
diagnostic scan does not mean its scientific gates passed.

The settled 40³, 0.6 fm checkpoint uses the same 24 fm box. It was prepared
on that mesh, rather than obtained by interpolating the coarse orbitals.
Its fresh residual meets the original 1e-6 MeV static criterion. The
z-aligned copy rotates both spatial indices and spinors, preserving the
discrete norm without interpolation. Changing only the input grid while
retaining a coarse checkpoint does not reproduce this test.

Long response spectra still require a matched unboosted control, full
endpoint conservation checks and timestep, mesh and box convergence.
Short-time overlap recovery cannot qualify the archived 6000 fm/c spectrum.
