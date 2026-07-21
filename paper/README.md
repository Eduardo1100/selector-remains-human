# Paper build

`main.md` is the sole editorial source for the manuscript. `main.tex` is a
generated derivative and must never be edited independently.

Build prerequisites:

- Pandoc, for Markdown-to-TeX conversion.
- Tectonic, or `latexmk` plus a PDF-capable TeX installation, for PDF output.
- Python dependencies from `pyproject.toml`, including Matplotlib, for figures.

Commands:

    make paper-assets   # validate tracked result rows and generate SVG/PNG/data
    make paper-check    # report missing document-build dependencies
    make paper-tex      # regenerate main.tex from main.md
    make paper          # generate main.pdf

The repository-local Mise toolchain pins Pandoc 3.10 and Tectonic 0.16.9.
Run mise install to provision those document tools, then run
mise exec -- make paper to generate the PDF. Keep mise.toml tracked as source
configuration.

The manuscript uses accessible SVG figure paths. High-resolution PNG and the
social-preview PNG are generated alongside them for publication systems that
cannot accept SVG.
