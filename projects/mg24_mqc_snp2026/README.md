# 24Mg monopole-quadrupole coupling

This isolated Sky3D v1.2 project tests deformation-induced
E0-E2(K=0) mixing in strongly prolate 24Mg with exactly two standard Skyrme
EDFs: **SLy5** and **SkM\***. The comparison is deliberately not a force
survey. It asks whether the off-diagonal TDHF signature survives a reasonable
change of EDF while allowing peak energies, centroids, fragmentation, and
strengths to change.

SV-bas results from the earlier project stage are retained for provenance but
are legacy-only. They are excluded from the new two-EDF tables, figures, and
manuscript conclusion. No SkP-delta calculation is in scope.

## Production design

Both EDFs use the same numerical protocol wherever stable:

- pairing: none;
- grid: 24 x 24 x 24, 1-fm spacing;
- production symmetry axis: z after an exact cubic-grid and spinor rotation;
- time step: 0.2 fm/c;
- propagation time: 18000 fm/c;
- protocol sampling: 2 fm/c;
- baseline boost: eta = 5e-5;
- E0 linearity boost: eta = 2.5e-5;
- primary window: exp[-Gamma_sm t/(2 hbar)];
- smoothing study: Gamma_sm = 1.0, 0.5, 0.2, 0.1, and 0.07 MeV;
- matched no-boost subtraction for every response channel.

The physical Rayleigh spacing is `2 pi hbar/T`; zero padding is used only to
interpolate the plotting grid and is never described as increased resolution.
The imposed Lorentzian smoothing width, finite-time spacing, and any physical
fragmentation are reported separately.

For each EDF the core calculation comprises:

1. static HF and the full duration-matched unperturbed TDHF trajectory;
2. E0 boost with Q00 and Q20 readout;
3. E2(K=0) boost with Q20 and Q00 readout;
4. half-strength E0 boost for linearity;
5. diagonal response, both signed off-diagonal responses, reciprocity,
   stability, smoothing, and duration-prefix convergence.

The high-resolution boundary audit uses density snapshots and a same-spacing
larger-box response calculation. Sky3D's wave-function derivatives are
periodic; `periodic=F` changes the Coulomb treatment but does not activate an
absorber. No mask or imaginary potential is added.

## Repository implementations

The unmodified local force table contains the production identifiers:

```text
SLy5  -> &force name='Sly5'
SkM*  -> &force name='SkMs'
```

The executable was compiled from checkout `be42efc`. Parameter blocks are
recorded in `Code/forces.data`; no interaction parameter is tuned or replaced.
The literature nuclear-matter values used for interpretation are stored in
`data/processed/edf_nuclear_matter_properties.csv` and `.json` with sources.

## Layout

- `CALCULATION_PLAN.md`: staged two-EDF execution and decision gates;
- `MANUSCRIPT_OUTLINE.md`: physics-led SNP2026 outline and figure plan;
- `configs/`: exact Sky3D namelists and analysis configurations;
- `runs/`: ignored wave functions, density snapshots, and raw protocols;
- `logs/`: WSL build and run provenance;
- `scripts/`: run, rotation, analysis, and plotting entry points;
- `data/processed/`: reproducible CSV/JSON/HDF5 products;
- `data/experimental/`: checked experimental inputs and provenance;
- `figures/`: vector PDF and PNG figures;
- `paper/`: manuscript source, which remains unchanged until the numerical
  convergence study is complete.

Run scripts refuse to overwrite non-empty case directories. All compilation
and execution is performed through WSL.

## WSL execution

From the checkout root:

```bash
cd /mnt/d/Coding/sky3d
export OMP_NUM_THREADS=18

# Build, if needed.
projects/mg24_mqc_snp2026/scripts/build_wsl.sh

# The current staged queue performs matched SLy5 and SkM* calculations.
projects/mg24_mqc_snp2026/scripts/run_two_edf_stage1_wsl.sh
```

The first stage completes the two full E0 responses and their matched
references. It is analyzed before reverse E2(K=0) and half-strength production
is released, matching the decision order in `CALCULATION_PLAN.md`. After the
documented E0 gate, `scripts/run_two_edf_stage2_wsl.sh` runs the two reverse
responses and the two half-strength checks. It refuses to start unless
`data/processed/highres/e0_stage_go.flag` contains exactly `GO`.

The two-EDF analyzer writes spectra, moments, EWSR exhaustion, stability,
duration convergence, reciprocity, and linearity to reproducible CSV/HDF5:

```bash
PYTHONPATH=/home/abhishek/.local/share/sky3d-response-deps:$PWD/Utils/Strength_Calculation \
python3 projects/mg24_mqc_snp2026/scripts/analyze_highres_convergence.py \
  --sly5-e0 projects/mg24_mqc_snp2026/runs/e0_sly5_t18000 \
  --sly5-reference projects/mg24_mqc_snp2026/runs/no_boost_sly5_t18000 \
  --skms-e0 projects/mg24_mqc_snp2026/runs/e0_skms_t18000 \
  --skms-reference projects/mg24_mqc_snp2026/runs/no_boost_skms_t18000
```

Once the reverse trajectories exist, `scripts/make_publication_figures.py`
generates the three requested figures and a compact four-panel alternative.
Every canvas is at most 6.35 inches wide; legends and panel labels are kept
short, and the extraction method is not printed inside the axes.

The separately gated `scripts/run_two_edf_box_check_wsl.sh` re-converges both
states on 32 x 32 x 32 grids and runs matched E0/reference pairs to 8000 fm/c.
The smaller and larger grids retain the same 1-fm spacing; the analyzer writes
their direct spectral mismatch to `data/processed/highres/box_size.csv`.

Response-package tests run in WSL with the isolated HDF5 dependency:

```bash
cd /mnt/d/Coding/sky3d/Utils/Strength_Calculation
export PYTHONPATH=/home/abhishek/.local/share/sky3d-response-deps:$PWD
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

## Interpretation boundary

The experimental alpha-scattering analyses measure isoscalar monopole
strength. The signed complex susceptibilities `chi_20,00` and `chi_00,20` are
theoretical TDHF diagnostics of mode mixing. A result may establish robustness
across SLy5 and SkM* only; it must not be generalized to EDF independence.

Detailed spectra are allowed to differ. The intended robust statement requires
both EDFs to show a prolate reference state, a low-energy monopole component,
coincident K=0 quadrupole strength, and a nonzero reciprocal off-diagonal
response in the same energy region. No energy shift or fitted normalization is
applied to either EDF.
