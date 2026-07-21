#!/usr/bin/env python3
"""Generate paper derivatives while keeping paper/main.md as the sole source."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SVG_PNG_SUBSTITUTIONS = {
    "(figures/selector_remains_human_figure1a.svg)": "(figures/selector_remains_human_figure1a.png)",
    "(figures/selector_remains_human_figure1b.svg)": "(figures/selector_remains_human_figure1b.png)",
}
PANEL_PNG_BLOCKS = {
    (
        "![Horizontal point-and-interval plot: conditional cross-predictability is strong, "
        "MiniLM centroid logistic is near chance, and directional MiniLM top-3 is stronger; "
        "vertical reference lines mark chance and the surface-format baseline.]"
        "(figures/selector_remains_human_figure1a.png)"
    ): (
        "\\begin{center}\n"
        "\\includegraphics[width=\\linewidth]{figures/selector_remains_human_figure1a.png}\n"
        "\\end{center}"
    ),
    (
        "![Horizontal point-and-interval plot: same-prompt preference labels create a strong "
        "transductive boundary, while random same-prompt domains and train- or test-labeled "
        "cross-prompt domains remain near chance; vertical reference lines mark chance and "
        "the surface-format baseline.](figures/selector_remains_human_figure1b.png)"
    ): (
        "\\begin{center}\n"
        "\\includegraphics[width=\\linewidth]{figures/selector_remains_human_figure1b.png}\n"
        "\\end{center}"
    ),
}
FIGURE_ONE_PANEL_PNGS = (
    "figures/selector_remains_human_figure1a.png",
    "figures/selector_remains_human_figure1b.png",
)
TABLE_2_TEX_START = "\\subsubsection{Table 2. Canonical LitBench mechanism"
TABLE_2_TEX_END = "\\subsection{7. Discussion}"
PAPER_TITLE = (
    "The Selector Remains Human: A Negative Result on Compression-Based Value Signals for Creative Text"
)
PDF_TITLE_BLOCK = (
    "\\begin{center}\n"
    "{\\large\\bfseries\\mbox{The Selector Remains Human}\\par}\n"
    "\\vspace{0.4em}\n"
    "{\\normalsize\\bfseries\\resizebox{\\linewidth}{!}{A Negative Result on Compression-Based Value Signals for Creative Text}\\par}\n"
    "\\end{center}"
)


def require(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise SystemExit(f"Missing required executable: {name}. See paper/README.md.")
    return executable


def select_pdf_engine(requested: str) -> tuple[str, str]:
    tectonic = shutil.which("tectonic")
    latexmk = shutil.which("latexmk")
    if requested == "tectonic":
        return "tectonic", require("tectonic")
    if requested == "latexmk":
        return "latexmk", require("latexmk")
    if tectonic:
        return "tectonic", tectonic
    if latexmk:
        return "latexmk", latexmk
    raise SystemExit(
        "Missing PDF engine: install tectonic, or install latexmk with a PDF-capable TeX installation."
    )


def temporary_pdf_markdown() -> Path:
    """Write a paper-local TeX input with only the two known SVG links replaced."""
    source = (PAPER / "main.md").read_text(encoding="utf-8")
    converted = source
    assert source.startswith(f"# {PAPER_TITLE}"), "Expected manuscript title missing from PDF build input."
    converted = converted.replace(f"# {PAPER_TITLE}", PDF_TITLE_BLOCK, 1)
    for svg_target, png_target in SVG_PNG_SUBSTITUTIONS.items():
        assert svg_target in converted, f"Expected Figure 1 SVG link missing: {svg_target}"
        converted = converted.replace(svg_target, png_target, 1)
    for markdown_block in PANEL_PNG_BLOCKS:
        assert markdown_block in converted, "Expected Figure 1 panel block missing from PDF build input."
    assert converted != source, "No Figure 1 SVG links were converted for PDF build."
    lines = converted.splitlines()
    normalized_lines: list[str] = []
    for index, line in enumerate(lines):
        if (
            not line
            and index > 0
            and index + 1 < len(lines)
            and lines[index - 1].lstrip().startswith("|")
            and lines[index + 1].lstrip().startswith("|")
        ):
            continue
        normalized_lines.append(line)
    converted = "\n".join(normalized_lines) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".md", prefix=".main_pdf_", dir=PAPER, delete=False
    ) as handle:
        handle.write(converted)
        return Path(handle.name)


def temporary_tex_header() -> Path:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tex", prefix=".pdf_header_", dir=PAPER, delete=False
    ) as handle:
        handle.write(
            "\\usepackage{pdflscape}\n"
            "\\AtBeginDocument{\\hypersetup{\n"
            f"  pdftitle={{{PAPER_TITLE}}},\n"
            "  pdfauthor={Eduardo Cortes},\n"
            "  pdfsubject={Compression-based value signals, contrastive domains, and human preference in creative text}\n"
            "}}\n"
        )
        return Path(handle.name)


def landscape_table_two(tex_path: Path) -> None:
    tex = tex_path.read_text(encoding="utf-8")
    start = tex.index(TABLE_2_TEX_START)
    end = tex.index(TABLE_2_TEX_END, start)
    section = tex[start:end]
    wrapped = "\\begin{landscape}\n\\scriptsize\n\\setlength{\\tabcolsep}{2pt}\n" + section
    wrapped += "\\end{landscape}\n\n"
    tex_path.write_text(tex[:start] + wrapped + tex[end:], encoding="utf-8")


def unnumber_figure_one_panels(tex_path: Path) -> None:
    """Turn only the two Figure 1 panels into centered, uncaptioned graphics."""
    tex = tex_path.read_text(encoding="utf-8")
    for png_target in FIGURE_ONE_PANEL_PNGS:
        graphic_position = tex.index(png_target)
        figure_start = tex.rfind("\\begin{figure}", 0, graphic_position)
        figure_end = tex.index("\\end{figure}", graphic_position) + len("\\end{figure}")
        assert figure_start >= 0, f"Expected Pandoc figure environment for {png_target}."
        figure_block = tex[figure_start:figure_end]
        assert figure_block.count("\\caption{") == 1, (
            f"Expected one Pandoc implicit caption for {png_target}."
        )
        image_end = figure_block.index("\\caption{")
        image_block = figure_block[len("\\begin{figure}\n\\centering\n"):image_end]
        assert png_target in image_block, f"Expected image missing from Figure 1 panel block: {png_target}."
        replacement = "\\begin{center}\n" + image_block + "\\end{center}"
        tex = tex[:figure_start] + replacement + tex[figure_end:]
    tex_path.write_text(tex, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Check requested paper-build dependencies.")
    parser.add_argument("--tex", action="store_true", help="Generate paper/main.tex from paper/main.md.")
    parser.add_argument("--pdf", action="store_true", help="Generate TeX and compile paper/main.pdf.")
    parser.add_argument("--engine", choices=["auto", "tectonic", "latexmk"], default="auto")
    args = parser.parse_args()
    if not (args.check or args.tex or args.pdf):
        parser.error("choose --check, --tex, or --pdf")
    pandoc = require("pandoc")
    engine_name, engine = select_pdf_engine(args.engine) if args.pdf else (None, None)
    print(f"pandoc: {pandoc}")
    if engine:
        print(f"{engine_name}: {engine}")
    if args.check:
        return
    temporary_input = temporary_pdf_markdown()
    temporary_header = temporary_tex_header()
    try:
        subprocess.run(
            [
                pandoc, str(temporary_input), "--from", "markdown+tex_math_dollars",
                "--standalone", "--to", "latex", "--include-in-header", str(temporary_header),
                "--output", str(PAPER / "main.tex"),
            ],
            check=True,
            cwd=ROOT,
        )
        unnumber_figure_one_panels(PAPER / "main.tex")
        landscape_table_two(PAPER / "main.tex")
        if engine_name == "tectonic":
            subprocess.run([engine, "--outdir", str(PAPER), str(PAPER / "main.tex")], check=True, cwd=ROOT)
        elif engine_name == "latexmk":
            subprocess.run([engine, "-pdf", "-interaction=nonstopmode", "main.tex"], check=True, cwd=PAPER)
    finally:
        temporary_input.unlink(missing_ok=True)
        temporary_header.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
