#!/usr/bin/env python3
"""Build Figure 1 from tracked LitBench summaries, never hand-entered results."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.fspath(Path(tempfile.gettempdir()) / "poemforge-mpl"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
matplotlib.rcParams["svg.hashsalt"] = "poemforge-primary-figure"
RESULTS = ROOT / "results" / "litbench"
OUTPUT = ROOT / "paper" / "figures"
SURFACE_ACCURACY = 0.6077922077922078
CHANCE_ACCURACY = 0.5


@dataclass(frozen=True)
class Selection:
    key: str
    panel: str
    label: str
    display_label: str
    source: str
    filters: dict[str, str]
    expected_accuracy: float
    operator: str
    label_source: str
    prompt_relation: str
    status: str


SELECTIONS = [
    Selection(
        "surface_baseline", "A", "Surface-format baseline", "Surface-format baseline",
        "litbench_prompt_domain_embedding_operator_models_test_embedding_operators_exact1155_mindomain2_maxdomain10.csv",
        {"operator": "mean", "model": "surface_format"}, SURFACE_ACCURACY,
        "5-fold surface logistic", "none", "n/a", "inductive feature baseline",
    ),
    Selection(
        "conditional_lm", "A", "Conditional cross-predictability",
        "Conditional cross-predictability\n(same prompt, preference labels)",
        "litbench_prompt_v_domain_contrast_models_test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10.csv",
        {"model": "domain_contrast_sign_rule"}, 0.7974025974025974,
        "conditional cross-predictability", "test preferences", "same prompt", "transductive",
    ),
    Selection(
        "minilm_centroid", "A", "MiniLM centroid logistic",
        "MiniLM centroid logistic\n(same prompt, preference labels)",
        "litbench_prompt_domain_kernel_models_test_prompt_domain_kernel_tfidf_embedding_mindomain2_maxdomain3.csv",
        {"kernel": "embedding", "model": "domain_specificity_logistic"}, 0.48917748917748916,
        "centroid logistic", "test preferences", "same prompt", "transductive control",
    ),
    Selection(
        "directional_top3", "A", "MiniLM directional top-3",
        "MiniLM directional top-3\n(same prompt, preference labels)",
        "litbench_prompt_domain_embedding_operator_models_test_embedding_operators_exact1155_mindomain2_maxdomain10.csv",
        {"operator": "top3", "model": "top3_domain_contrast_sign_rule"}, 0.8770562770562771,
        "directional top-3", "test preferences", "same prompt", "transductive probe",
    ),
    Selection(
        "same_prompt_labels", "B", "Same-prompt preference labels",
        "Same-prompt preference labels\nMiniLM top-3, transductive",
        "litbench_prompt_domain_embedding_operator_models_test_embedding_operators_exact1155_mindomain2_maxdomain10.csv",
        {"operator": "top3", "model": "top3_domain_contrast_sign_rule"}, 0.8770562770562771,
        "MiniLM top-3, transductive", "test preferences", "same prompt", "transductive probe",
    ),
    Selection(
        "same_prompt_random", "B", "Same-prompt random domains",
        "Same-prompt random domains\nconditional cross-predictability",
        "litbench_prompt_v_random_split_models_test_prompt_v_distilgpt2_randomsplit_seed123_mindomain2_maxdomain3.csv",
        {"model": "random_domain_contrast_sign_rule"}, 0.4892334194659776,
        "conditional cross-predictability", "random split", "same prompt", "prompt-only control",
    ),
    Selection(
        "cross_prompt_train", "B", "Cross-prompt train labels",
        "Cross-prompt train labels\nMiniLM top-5",
        "litbench_train_domain_embedding_models_train_domain_embedding_exact1155_min2_max10_top50.csv",
        {"operator": "top5", "model": "top5_train_domain_sign_rule"}, 0.5307359307359307,
        "MiniLM top-5", "train preferences", "different/similar prompts", "inductive cross-prompt",
    ),
    Selection(
        "cross_prompt_test", "B", "Cross-prompt test labels",
        "Cross-prompt test labels\nMiniLM mean",
        "litbench_cross_prompt_test_domain_embedding_models_cross_prompt_test_domain_embedding_exact1155_min2_max10_top50.csv",
        {"operator": "mean", "model": "mean_cross_prompt_test_domain_sign_rule"}, 0.5298701298701298,
        "MiniLM mean", "test preferences", "different/similar prompts", "transductive cross-prompt",
    ),
]


def select_row(selection: Selection) -> dict[str, object]:
    path = RESULTS / selection.source
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if all(row.get(column) == value for column, value in selection.filters.items())
        ]
    assert len(rows) == 1, f"{selection.key}: expected one row in {path}, found {len(rows)}"
    row = rows[0]
    accuracy = float(row["accuracy"])
    assert abs(accuracy - selection.expected_accuracy) < 1e-12, (
        f"{selection.key}: expected {selection.expected_accuracy}, got {accuracy}"
    )
    return {
        "key": selection.key,
        "panel": selection.panel,
        "label": selection.label,
        "display_label": selection.display_label,
        "source_path": str(Path("results/litbench") / selection.source),
        "filters": selection.filters,
        "accuracy": accuracy,
        "ci95_low": float(row["ci95_low"]),
        "ci95_high": float(row["ci95_high"]),
        "n": int(row["n"]),
        "operator": selection.operator,
        "label_source": selection.label_source,
        "prompt_relation": selection.prompt_relation,
        "transductive_inductive_status": selection.status,
    }


def accessible_svg(path: Path, title: str, description: str) -> None:
    text = path.read_text(encoding="utf-8")
    svg_start = text.find("<svg")
    marker = text.find(">", svg_start)
    assert svg_start != -1 and marker != -1, f"Unexpected SVG: {path}"
    replacement = (
        ' role="img" aria-labelledby="figure-title figure-desc">'
        f'<title id="figure-title">{title}</title>'
        f'<desc id="figure-desc">{description}</desc>'
    )
    path.write_text(text[:marker] + replacement + text[marker + 1 :], encoding="utf-8")
    normalize_svg(path)


def normalize_svg(path: Path) -> None:
    """Remove trailing spaces and tabs while preserving each SVG line ending."""
    text = path.read_text(encoding="utf-8")
    normalized_lines = []
    for line in text.splitlines(keepends=True):
        if line.endswith("\r\n"):
            content, ending = line[:-2], "\r\n"
        elif line.endswith("\n") or line.endswith("\r"):
            content, ending = line[:-1], line[-1]
        else:
            content, ending = line, ""
        normalized_lines.append(content.rstrip(" \t") + ending)
    path.write_text("".join(normalized_lines), encoding="utf-8")


def panel_description(rows: list[dict[str, object]], mechanism: str) -> str:
    values = "; ".join(
        f"{row['label']}: {float(row['accuracy']) * 100:.1f}% "
        f"(95% CI {float(row['ci95_low']) * 100:.1f}–{float(row['ci95_high']) * 100:.1f}; n={row['n']})"
        for row in rows
    )
    return (
        f"Horizontal point-and-interval plot. Accuracy is on the horizontal axis; "
        f"the dashed vertical line marks chance at 50.0% and the dotted vertical line "
        f"marks the 60.8% surface-format baseline. {mechanism} {values}"
    )


def draw_panel(rows: list[dict[str, object]], panel: str, output_stem: str) -> None:
    title = (
        "A. Representation is not the mechanism; the directional operator is"
        if panel == "A"
        else "B. The strong boundary requires prompt-local preference labels"
    )
    fig, ax = plt.subplots(figsize=(14, 9), constrained_layout=True)
    positions = list(range(len(rows)))
    values = [float(row["accuracy"]) for row in rows]
    lower = [value - float(row["ci95_low"]) for value, row in zip(values, rows)]
    upper = [float(row["ci95_high"]) - value for value, row in zip(values, rows)]

    ax.errorbar(
        values, positions, xerr=[lower, upper], fmt="o", markersize=10,
        color="#1F77B4", ecolor="#1F77B4", elinewidth=2.2, capsize=7, capthick=2.2,
        zorder=3,
    )
    ax.axvline(CHANCE_ACCURACY, color="#1F77B4", linewidth=1.8, linestyle="--", zorder=1)
    ax.axvline(SURFACE_ACCURACY, color="#1F77B4", linewidth=1.8, linestyle=":", zorder=1)
    ax.set_xlim(0.43, 0.98)
    ax.set_ylim(len(rows) - 0.65, -0.18)
    ax.set_yticks(positions, [str(row["display_label"]) for row in rows], fontsize=15)
    ax.set_xlabel("Pairwise accuracy (95% bootstrap interval)", fontsize=16)
    ax.set_title(title, loc="left", fontsize=20, fontweight="bold", pad=14)
    ax.grid(axis="x", color="#D9D9D9", linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for value, position, row in zip(values, positions, rows):
        ax.annotate(
            f"{value:.1%}  n={row['n']}", xy=(value, position), xytext=(12, 0),
            textcoords="offset points", va="center", ha="left", fontsize=14,
            color="#111111",
        )

    svg = OUTPUT / f"{output_stem}.svg"
    png = OUTPUT / f"{output_stem}.png"
    fig.savefig(
        svg,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.16,
        metadata={"Date": None},
    )
    normalize_svg(svg)
    fig.savefig(png, dpi=400, bbox_inches="tight", pad_inches=0.16)
    plt.close(fig)
    mechanism = (
        "The directional MiniLM top-3 operator exceeds the conditional language-model readout, "
        "while centroid pooling is near chance."
        if panel == "A"
        else "Only same-prompt preference labels produce the strong transductive boundary; "
        "random or cross-prompt domains remain near chance."
    )
    accessible_svg(svg, title, panel_description(rows, mechanism))


def draw_social(rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(12, 6.3), constrained_layout=True)
    positions = list(range(len(rows)))
    values = [float(row["accuracy"]) for row in rows]
    lower = [value - float(row["ci95_low"]) for value, row in zip(values, rows)]
    upper = [float(row["ci95_high"]) - value for value, row in zip(values, rows)]
    ax.errorbar(values, positions, xerr=[lower, upper], fmt="o", markersize=10, color="#1F77B4", capsize=6)
    ax.axvline(CHANCE_ACCURACY, color="#4D4D4D", linewidth=1.4, linestyle="--")
    ax.axvline(SURFACE_ACCURACY, color="#4D4D4D", linewidth=1.4, linestyle=":")
    ax.set_xlim(0.43, 0.98)
    ax.set_ylim(len(rows) - 0.6, -0.25)
    ax.set_yticks(positions, [str(row["display_label"]) for row in rows], fontsize=11)
    ax.set_xlabel("Pairwise accuracy (95% bootstrap interval)")
    ax.set_title("The Selector Remains Human", fontsize=22, fontweight="bold", loc="left")
    ax.grid(axis="x", color="#D9D9D9")
    ax.set_axisbelow(True)
    fig.savefig(OUTPUT / "selector_remains_human_social_preview.png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate source selections without writing files.")
    args = parser.parse_args()
    rows = [select_row(selection) for selection in SELECTIONS]
    if args.check:
        print("Primary-figure source selections verified.")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "purpose": "Canonical Figure 1 data selected from tracked LitBench summaries.",
        "canonical_surface_note": (
            "The surface baseline is selected from the exact-overlap embedding-operator "
            "summary (0.607792). The conditional-LM summary's 0.602597 surface row is "
            "a separate cross-validation run and is not interchangeable."
        ),
        "rows": rows,
    }
    (OUTPUT / "figure1_data.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    draw_panel([row for row in rows if row["panel"] == "A"], "A", "selector_remains_human_figure1a")
    panel_b = [row for row in rows if row["panel"] == "B"]
    draw_panel(panel_b, "B", "selector_remains_human_figure1b")
    draw_social(panel_b)
    print(f"Wrote figure data and visual assets to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
