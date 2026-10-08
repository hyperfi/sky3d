# Experimental-data provenance

Search completed on 2026-08-29. The APS article records, arXiv abstracts and
ancillary source bundles, Zenodo, and HEPData were checked before any
digitization. No publisher- or author-supplied numerical file containing the
full 9-25 MeV 24Mg IS0 distribution was found. Neither arXiv bundle contains
CSV, table, or analysis-data files; the 2022 bundle contains TeX plus EPS
figures and the 2026 bundle contains TeX plus PDF figures.

## Primary publications

- A. Bahini et al., *Phys. Rev. C* **105**, 024311 (2022),
  DOI `10.1103/PhysRevC.105.024311`, arXiv `2111.07105`.
- A. Bahini et al., *Phys. Rev. C* **113**, 064313 (2026),
  DOI `10.1103/29j9-j57p`, arXiv `2511.21880`.

The 2022 paper states that the iThemba distribution uses 0.5-MeV bins and that
the displayed uncertainties include statistical and systematic components.
Its Table III supplies numerical energies, EWSR fractions, and IS0 strengths
for resolved prominent 0+ states through 15.78 MeV; these values are transcribed
without digitization into `prominent_states_bahini2022.csv`.

The 2026 paper reuses the same iThemba experimental strength at 70-keV FWHM
for a continuous-wavelet analysis. Its Tables III-V supply characteristic
scales for 10-24, 10-18 (MQC), and 18-24 MeV (ISGMR); these are transcribed
without digitization into `wavelet_scales_bahini2026.csv`.

## Full distribution: explicitly digitized

The complete binned distribution exists only in figures. The preserved source
`source/fig5_bahini2022_arxiv.eps` is the author-supplied Cairo vector figure
from the arXiv `2111.07105` source archive (SHA-256
`92009487c140d8b8169de8fe3810f281619df0dfc014f021bb6dab77ae3fdba3`).

`scripts/digitize_bahini_2022_eps.py` parses the upper-panel black-circle
centers and error-bar endpoints directly from that vector file. It calibrates
against the encoded major ticks, checks the source hash, requires exactly 31
half-MeV bins from 9.75 to 24.75 MeV, and writes
`digitized_is0_bahini2022.csv` plus JSON metadata. The 13.75-MeV bin integrates
to 37.24 fm4, independently agreeing with the tabulated 13.87-MeV state value
37.7(3.8) fm4.

The coordinate-reading uncertainty is estimated conservatively as
0.2 fm4/MeV (approximately half a plotted line width) and is stored separately
from the experimental error bars. These data must always be labeled
"digitized from the author arXiv vector figure", never "publisher-supplied
numerical data".

## Additional source checked

Armand Bahini's 2021 Wits PhD dissertation, handle `10539/32505`, documents
the DoS-to-EWSR-to-strength conversion, the 500-keV 24Mg rebinning, and error
propagation, but likewise presents the full distribution only graphically.
