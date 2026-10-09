# Sky3D single-GPU technical report

The standalone report documents source snapshot `9576125`, implementation
changes, strict numerical controls, failed physical gates, complete-job
benchmarks, response plots, memory policy and the new-clone workflow.
It does not rerun nuclear simulations or replace saved timing evidence.

From Linux/WSL, compile the retained source, vector figures and tables with
an existing PDFLaTeX installation:

```bash
python3 docs/report/build_report.py
```

The PDF is written to `output/pdf/sky3d_single_gpu_report.pdf` and build
intermediates stay in the ignored `docs/report/build/` directory. The builder
also recognizes an existing Windows PDFLaTeX executable via WSL interop.
It installs no packages or compiler. The report uses common LaTeX packages
including Latin Modern, geometry, amsmath, graphicx, booktabs, longtable,
listings, xcolor, hyperref, fancyhdr and placeins.

`make_assets.py` regenerates figures and tables using saved JSON reports and
the preserved full response time histories on the original evidence host.
It requires NumPy/Matplotlib and the raw cache files identified in the report;
it verifies the recorded CPU/GPU spectral difference before exporting figures.
Ordinary PDF compilation uses retained figures and needs none of those large
raw scientific files. `evidence_manifest.json` records exact evidence, source
and response-input hashes.

The final PDF is checked for unresolved references, font embedding, text
extraction and rendered page layout. Its scientific conclusions remain those
of the cited saved evidence, including the failed coarse long-response
orthogonality gate.
