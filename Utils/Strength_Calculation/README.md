# Sky3D response analysis

`Fourier.py` is now a configuration-driven analysis entry point rather than an
interactive one-off script. It reads Sky3D `*.res` time signals and produces:

- signed diagonal and off-diagonal response densities;
- declared electric `dB(Elambda)/dE` distributions;
- E1 photoabsorption cross sections and dipole polarizability;
- `m_-1`, `m_0`, `m_1`, and `m_1/m_0` in named energy intervals;
- EWSR exhaustion against either an explicit reference or the E1 TRK value;
- observed FWHM, imposed artificial smoothing FWHM, and a separately labeled
  Lorentzian deconvolution estimate;
- long-form CSV, structured HDF5, metadata JSON, and vector PDF/PNG plots.

The package does not modify Sky3D or infer electromagnetic normalization from a
filename. A channel is labeled `B(Elambda)` only after the configuration
explicitly declares `type = "electric"`, `diagonal = true`, and
`b_elambda = true`.

## WSL setup and use

Run installation, tests, and analysis inside WSL:

```bash
cd /mnt/d/Coding/sky3d/Utils/Strength_Calculation
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e .
python -m unittest discover -s tests -v
python Fourier.py inspect --input ../../Test/GR/for005.gr
python Fourier.py analyze --config examples/synthetic_e1.toml
```

On this WSL host, `python3-venv` is not installed and the base interpreter is
PEP-668 managed. The non-invasive fallback used for verification was:

```bash
python3 -m pip install --target \
  /home/abhishek/.local/share/sky3d-response-deps --no-deps 'h5py>=3.8'
export PYTHONPATH=/home/abhishek/.local/share/sky3d-response-deps:$PWD
python3 -m unittest discover -s tests -v
```

NumPy and Matplotlib were already supplied by WSL. Use a venv, or install all
of `requirements.txt` into the target directory, on a clean machine.

Generate the analytic demonstration input first:

```bash
python examples/make_synthetic_e1_run.py
```

The demonstration is a normalization test, not a Sky3D prediction. For a real
run, copy `examples/synthetic_e1.toml`, replace the run path and channel list,
and keep the configuration beside the resulting data.

For a scratch validation that leaves the checkout untouched:

```bash
python examples/make_synthetic_e1_run.py --output /tmp/sky3d-synthetic-e1
python Fourier.py analyze --config examples/synthetic_e1.toml \
  --run-directory /tmp/sky3d-synthetic-e1 \
  --output /tmp/sky3d-synthetic-e1-output
```

## Sky3D conventions represented here

This checkout identifies itself as Sky3D v1.2. `Code/external.f90` applies

```text
psi(0+) = exp(-i eta F) psi(0).
```

For `L >= 2`, its implemented multipole is
`sqrt(2L+1) r^L Y_LM`. The monopole special case is
`r^2/sqrt(4 pi)`, not `Y_00` times `r^0`. The isoscalar dipole has a separate
center-of-mass-corrected radial form. The response files contain time in the
first column and normally IS and IV observables in columns 1 and 2,
respectively (using zero-based numeric column indices in the TOML file).
Sky3D appends the restart point once when a continuation begins. The reader
removes only an adjacent, numerically consistent duplicate; it rejects a
duplicate time carrying inconsistent observables.

For an electric same-operator response produced with `only_P=1`, analyze the
applied-field expectation in `extfield.res` unless the normalization of another
readout has been independently established. Deformed fixed-K components also
need the appropriate laboratory-frame reconstruction before they are described
as a total experimental `B(Elambda)` distribution.

## Transform and normalization

For a sampled observable `A(t)`, the code removes the configured initial
baseline, applies a time window, and evaluates

```text
I_AF(E) = integral_0^T dt exp(-i E t / hbar) delta<A>(t).
```

If the boost is `exp(i s eta F)`, where Sky3D has `s=-1`, the complex response
density and signed spectral density are

```text
R_AF(E) = -s I_AF(E)/(eta hbar),
S_AF(E) = Im R_AF(E)/pi.
```

The `dt` time-integration factor is included. The previous script multiplied
NumPy's FFT by two but omitted `dt`, giving a normalization dependent on the
output sampling interval. The present convention is tested against an analytic
single mode whose integrated strength is known.

For a diagonal response, `S_FF(E)` is positive in the ideal linear-response
limit. For a cross response, `S_AF(E)` is a signed transition-product density;
the package retains its real response, imaginary response, and sign. It never
replaces a cross response by an absolute value.

## Windows, resolution, and widths

An exponential window uses

```text
w(t) = exp[-Gamma_sm t/(2 hbar)].
```

It therefore introduces a Lorentzian artificial FWHM exactly equal to
`Gamma_sm`. The summary reports:

1. `observed_fwhm_mev` from the plotted, smoothed dominant peak;
2. `artificial_smoothing_fwhm_mev = Gamma_sm`;
3. `intrinsic_fwhm_lorentzian_estimate_mev = observed - Gamma_sm`.

The third number is only a line-shape-dependent estimate, not an automatic
physical spreading width. A cosine-power window is supported for comparison,
but it has no unique Lorentzian FWHM, so the last two fields are `nan` rather
than being assigned a misleading width.

The metadata distinguishes the energy-bin spacing from the physical Rayleigh
resolution,

```text
Delta E_Rayleigh = 2 pi hbar / T.
```

Zero padding reduces the plotted bin spacing but does not improve this
resolution.

## Derived electric-dipole quantities

For a confirmed electric E1 channel, the package identifies the signed strength
with `dB(E1)/dE` and calculates

```text
sigma_gamma(E) [mb] = 4.022... E [MeV] dB(E1)/dE [e^2 fm^2/MeV],
alpha_D [fm^3] = (8 pi/9) e^2 m_-1,
m_k = integral_(Emin)^(Emax) E^k S(E) dE.
```

The integration interval is always present in `summary.csv`. Strength is not
silently clipped at zero; `negative_area_fraction` records numerical ringing or
other sign contamination in a nominally diagonal channel.

Set either `ewsr_reference` and `ewsr_label` on a channel, or use
`use_trk_ewsr = true` for electric E1 with `[nucleus] A` and `Z`. The latter
uses

```text
m1(TRK) = (9/4 pi) (hbar^2/2m_N) (N Z/A) (1+kappa).
```

Other multipoles and isoscalar operators require an explicit EWSR reference
consistent with the exact operator normalization used in the run.

## Configuration keys

Each `[[channels]]` table selects a file and observable column. Important keys
are:

- `diagonal`: `false` for an off-diagonal readout;
- `boost_amplitude`: optional override of `&extern ampl_ext`;
- `boost_phase_sign`: `-1` for this Sky3D implementation;
- `observable_scale` and `boost_operator_scale`: explicit normalization
  conversions when a documented operator conversion is required;
- `reference_file`, `reference_column`, and `reference_scale`: optional matched
  no-boost signal subtraction before windowing and transformation; the source
  and reference time grids must match exactly;
- `strength_units`: stored verbatim in CSV/HDF5 and used on plots;
- `b_elambda`: opt-in electric labeling;
- `ewsr_reference` or `use_trk_ewsr`: denominator for EWSR exhaustion.

Named `[[regions]]` produce separate moment, centroid, width, polarizability,
and sum-rule rows. Named `[[windows]]` make smoothing-dependence comparisons
reproducible in a single run.

For long-time boundary checks, `read_tdd_density` reads the total density from
Sky3D unformatted `*.tdd` snapshots. `density_diagnostics` returns the
density-integrated particle number, rms radius, center of mass, population in
a declared outer shell, and density on the outermost grid layer. These are
spatial diagnostics; they do not substitute for a larger-box response test.

## Output schema

- `spectra.csv`: configured energy range, complex response components, signed
  strength, optional `dB/dE`, and optional photoabsorption;
- `time_signals.csv`: raw signal, optional reference signal and reference delta,
  corrected baseline-subtracted signal, window, and filtered signal;
- `summary.csv`: moments and all integrated observables by channel/window/region;
- `response.h5`: full Nyquist-range spectra plus signals, configuration, hashes,
  conventions, and summaries;
- `metadata.json`: code commit, input/config hashes, constants, software versions,
  input namelists, and formulas;
- `spectrum_<channel>.pdf/.png` and `spectra_overview.pdf/.png`: regenerated
  publication plots.
