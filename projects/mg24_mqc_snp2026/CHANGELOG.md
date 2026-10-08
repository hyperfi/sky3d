# Final manuscript audit

## 2026-08-30

- Locked the calculation scope. No new Sky3D trajectory was started or rerun;
  all changes use the completed 24-cubed two-EDF data and the completed SLy5
  24-cubed versus 32-cubed check.
- Made the deformation convention explicit. The manuscript now quotes the
  total-mass moments `Q20 = 36.532 fm^2` (SLy5) and `35.568 fm^2` (SkM*) and
  the Sky3D definition `beta2 = 4*pi*Q20/(5*A*R^2)`, with
  `R = 1.2*A^(1/3) fm`. These give 0.319 and 0.311. They are not presented as
  charge-deformation values or compared numerically with values using an
  unspecified radius or charge convention.
- Replaced the ambiguous cross-strength notation by the computed quantity:
  `R_BA(E) = Im chi_BA^(-)(E)/pi`, using Sky3D's negative-exponent Fourier
  convention. Diagonal `R_AA` is denoted by `S_AA`; off-diagonal `R_BA` is a
  signed mixed response and is not positive definite.
- Stated the response units (`fm^4 MeV^-1`) and warned that the `F00` and
  `F20` operator normalizations differ, so absolute heights in different
  panels should not be compared directly.
- Corrected the reciprocal complex-response mismatches from the saved CSV to
  0.108% (SLy5) and 0.097% (SkM*) at 0.2-MeV artificial smoothing. Added the
  Onsager--Kubo condition for two time-even Hermitian operators in a
  time-reversal-invariant reference state.
- Added theory-minus-experiment centroid shifts: `(0.627, 0.304) MeV` for
  SLy5 and `(0.416, 0.452) MeV` for SkM* in the 10--18 and 18--25 MeV windows.
- Recast the box check as a convergence hierarchy. Window centroids and
  integrals are stable, whereas the narrowly smoothed pointwise spectrum and
  intrinsic widths are not box converged. The signed mixed-response integral
  is used only as a numerical stability diagnostic, not as a coupling metric.
- Removed the phrase "high-resolution TDHF". The high-resolution wording is
  retained only for the experimental alpha-scattering measurement.
- Kept Figure 2 sparse: compact mathematical axis labels, no extra annotations,
  and vertically stacked legends inside the upper-right corner of panels (c)
  and (d).
- Kept five numbered references by grouping the two Bahini papers and the two
  original EDF papers. The user's 2024 Sky3D/TDHF paper remains a direct method
  citation, and Kubo's linear-response paper is added for reciprocity.

### Final validation

- Regenerated both publication figures from the existing `spectra.csv`; no
  response or TDHF calculation was rerun.
- The response-package test suite passes: `10 passed` with the repository
  package supplied through `PYTHONPATH`.
- Two PdfLaTeX passes were launched from WSL. The installed PDF has exactly two
  A4 pages, and the final log contains no overfull boxes, undefined citations,
  or undefined cross-references.
- All fonts are embedded and no Type 3 font is present. `pdfimages -list`
  reports no raster image objects, preserving the vector publication figures.
- Both pages were rendered at 180 dpi and inspected. Text, captions, axes, and
  legends remain inside the page and figure bounds; a word-bounding-box check
  also places all extracted text within the A4 media boxes.
- Final PDF SHA-256:
  `709e47ceda338c404c930ba3807ef8f01aea60da2b924354682db5ea4ff435bb`.

## Final language and legibility pass

- Replaced “measure the coupling” by “directly expose the dynamical
  correlation” and removed the redundant “real-time” before TDHF.
- Clarified the Conclusion antecedent by naming “this induced motion” rather
  than using the ambiguous pronoun “it”.
- Rebuilt Figure 1 from the unchanged CSV data at near-column dimensions, with
  larger effective text, heavier theory curves and error bars, and a compact
  vertically stacked legend in unused upper-right space. Figure 2 and all
  numerical data are unchanged.
- Applied the no-ai-slop checks with minimum edits: no added claim, no inflated
  emphasis, no generic filler, and no change to the direct physics-first voice.
- The final typo, notation, and reference audit found five cited and defined
  bibliography keys, no uncited entry, no missing figure label, and consistent
  `00/20` response indices. Both centroid-shift pairs were rederived from the
  saved comparison JSON and reproduce the manuscript values.
