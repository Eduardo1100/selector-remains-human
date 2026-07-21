# Publication bundle

Make publication-bundle builds site/dist, a self-contained static bundle. It
contains the essay source, manuscript source, Figure 1 SVG and PNG assets,
social preview, provenance manifest, citation metadata, reproducibility notes,
license notices, the working-paper PDF, and an integration README.

Source-controlled inputs:

- paper/main.md, paper/main.pdf, essay.md, CITATION.cff, license files, site
  source, and documentation;
- tracked LitBench summaries selected by scripts/make_primary_figure.py.

Generated files:

- paper/figures/figure1_data.json and the Figure 1 SVG/PNG assets;
- site/dist/, rebuilt by make site.

Before publication:

1. Run make publication-check.
2. Run make paper when Pandoc and a PDF engine are available; inspect the PDF.
3. Confirm the bundled relative PDF link works from site/dist/paper/main.pdf.
4. Confirm the [software](../LICENSE) and [content](../LICENSE-CONTENT.md)
   license notices have the intended scope. Third-party datasets, frozen
   external artifacts, model outputs, and material carrying its own terms are
   not relicensed.
5. Obtain Eduardo's approval before staging, committing, pushing, publishing,
   or changing repository visibility.
