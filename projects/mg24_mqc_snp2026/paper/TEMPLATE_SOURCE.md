# Official template provenance

- official page: `https://www.sympnp.org/snp2026/`, Submissions -> Templates;
- official bundle: `https://www.sympnp.org/docs/pdflatex_template.zip`;
- retrieved: 2026-08-29;
- bundle SHA-256: `24e63c8c3548a800bfc2f230caf8be97c49594cf890b983a160772ef9d4377c8`;
- official `snp-official.cls` SHA-256: `309eedef82a8b4d22858f7dab26b92ed7106e83a03d61857aedddc5f481db63d`;
- official sample SHA-256: `392391b5e721f38d099f61e5e2b382ac436ea468111792f4dbf8caa66abc472d`.

The bundle was listed by the live 2026 site specifically as the PdfLaTeX
template for contributory, thesis, and Young Achiever Award manuscripts.
`official_cont_samp.tex` is preserved unmodified. `main.tex` follows its class,
page geometry, two-column layout, and camera-ready style.

The bundled class is REVTeX 4.0 from 2001 and its legacy `\document` patch is
incompatible with LaTeX kernels from 2020-10-01 onward. Following REVTeX's own
compatibility update, the working `snp.cls` places the same initialization in
the supported `begindocument/before` hook. `snp-official.cls` preserves the
downloaded bytes for audit; no layout definition is changed.
