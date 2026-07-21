from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ANALYSES_DIR = ROOT / "results" / "analyses"
PHASE_A_RESULTS = ROOT / "results" / "phase_a_eval_results"
TABLE_DIR = ROOT / "results" / "tables"
HASH_DIR = ROOT / "results" / "hashes"

READOUT_CONVERGENCE = ANALYSES_DIR / "readout_convergence_summary.csv"
BOOTSTRAP_TABLE = ANALYSES_DIR / "bootstrap_manuscript_table.csv"
GENERIC_D = ANALYSES_DIR / "generic_d_correlation_summaries.csv"
ITEM_NLL = PHASE_A_RESULTS / "item_nll_target_correlations.csv"
ABSOLUTE_EFFECTS = ANALYSES_DIR / "bootstrap_absolute_effects_summary.csv"
DOMAIN_CONTRAST_SURPRISE = ANALYSES_DIR / "bootstrap_domain_contrast_summary.csv"
GENERIC_ABSOLUTE_SURPRISE = ANALYSES_DIR / "bootstrap_generic_absolute_summary.csv"


FEATURE_SET_LABELS = {
    "none": "No residual controls",
    "other_human_targets": "Other human targets",
    "other_human_plus_surface": "Other human targets + surface",
    "stacked": "Other human + surface + NLL",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_shape(path: Path) -> tuple[int | None, int | None]:
    if path.suffix.lower() != ".csv":
        return None, None
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return 0, 0
        return sum(1 for _ in reader), len(header)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")


def fmt_float(x, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):.{digits}f}"


def fmt_signed(x, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):+.{digits}f}"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a simple GitHub-style markdown table without tabulate."""
    cols = [str(c) for c in df.columns]

    def cell(x) -> str:
        if pd.isna(x):
            return ""
        text = str(x)
        text = text.replace("\n", " ")
        text = text.replace("|", "\\|")
        return text

    rows = [[cell(v) for v in row] for row in df.to_numpy()]
    widths = []
    for i, col in enumerate(cols):
        values = [r[i] for r in rows]
        widths.append(max([len(col), *(len(v) for v in values)]))

    def fmt_row(values: list[str]) -> str:
        padded = [v.ljust(widths[i]) for i, v in enumerate(values)]
        return "| " + " | ".join(padded) + " |"

    header = fmt_row(cols)
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = [fmt_row(r) for r in rows]
    return "\n".join([header, sep, *body])


def write_table_bundle(df: pd.DataFrame, stem: str, caption: str) -> list[Path]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    tex_path = TABLE_DIR / f"{stem}.tex"

    df.to_csv(csv_path, index=False)

    md = dataframe_to_markdown(df)
    md_path.write_text(md + "\n", encoding="utf-8")

    tex = df.to_latex(
        index=False,
        escape=True,
        caption=caption,
        label=f"tab:{stem}",
    )
    tex_path.write_text(tex, encoding="utf-8")

    return [csv_path, md_path, tex_path]


def make_table_1_readout_convergence() -> list[Path]:
    require(READOUT_CONVERGENCE)
    df = pd.read_csv(READOUT_CONVERGENCE)

    # Prefer the manuscript-relevant order.
    order = ["none", "other_human_targets", "other_human_plus_surface"]
    df["feature_order"] = df["feature_set"].map({v: i for i, v in enumerate(order)}).fillna(99)
    df = df.sort_values("feature_order")

    out = pd.DataFrame(
        {
            "Controls": df["feature_set"].map(FEATURE_SET_LABELS).fillna(df["feature_set"]),
            "Compression": df["compression"].map(fmt_float),
            "Embedding": df["embedding"].map(fmt_float),
            "TF-IDF": df["tfidf"].map(fmt_float),
            "Compression - Embedding": df["compression_minus_embedding"].map(fmt_signed),
            "Compression - TF-IDF": df["compression_minus_tfidf"].map(fmt_signed),
        }
    )

    return write_table_bundle(
        out,
        "table_1_readout_convergence",
        "Mean run-level Spearman correlations for structural preference readouts across controls.",
    )


def make_table_2_bootstrap_uncertainty() -> list[Path]:
    require(BOOTSTRAP_TABLE)
    df = pd.read_csv(BOOTSTRAP_TABLE)

    out = pd.DataFrame(
        {
            "Claim": df["claim_id"],
            "Comparison": df["comparison"],
            "Observed Δ": df["observed_diff"].map(fmt_signed),
            "95% CI": df["ci95"],
            "Bootstrap n": df["n_boot"],
            "Interpretation": df["paper_interpretation"],
        }
    )

    return write_table_bundle(
        out,
        "table_2_bootstrap_uncertainty",
        "Poem-level bootstrap uncertainty for compression and matched-other comparisons.",
    )


def make_table_3_generic_d_summary() -> list[Path]:
    require(GENERIC_D)
    df = pd.read_csv(GENERIC_D)

    # The generic-D file uses Phase A correlation summaries.
    # Expected columns: metric,spearman_rho,p_value,source_file,domain_variant,control_variant,seed
    needed = {"metric", "spearman_rho", "domain_variant", "control_variant"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"generic_d table missing columns: {sorted(missing)}")

    sub = df[df["metric"].isin(["v_raw", "v_ctrl", "v_struct"])].copy()

    grouped = (
        sub.groupby(["domain_variant", "control_variant", "metric"], dropna=False)
        .agg(
            mean_rho=("spearman_rho", "mean"),
            min_rho=("spearman_rho", "min"),
            max_rho=("spearman_rho", "max"),
            n=("spearman_rho", "count"),
        )
        .reset_index()
        .sort_values(["domain_variant", "control_variant", "metric"])
    )

    out = pd.DataFrame(
        {
            "Domain": grouped["domain_variant"],
            "Control": grouped["control_variant"],
            "Metric": grouped["metric"],
            "Mean ρ": grouped["mean_rho"].map(fmt_signed),
            "Range ρ": [
                f"[{fmt_signed(lo)}, {fmt_signed(hi)}]"
                for lo, hi in zip(grouped["min_rho"], grouped["max_rho"])
            ],
            "n": grouped["n"],
        }
    )

    return write_table_bundle(
        out,
        "table_3_generic_d_summary",
        "Generic Gutenberg-domain compression correlations with human Surprise ratings.",
    )


def make_table_4_item_nll_correlations() -> list[Path]:
    require(ITEM_NLL)
    df = pd.read_csv(ITEM_NLL)

    # Keep this flexible because Phase A column names may vary slightly.
    model_col = next((c for c in df.columns if c.lower() in {"model", "lm", "model_name"}), None)
    target_col = next((c for c in df.columns if "target" in c.lower()), None)
    rho_col = next((c for c in df.columns if "rho" in c.lower() or "spearman" in c.lower()), None)
    p_col = next((c for c in df.columns if c.lower() in {"p", "p_value", "pvalue"}), None)

    if not model_col or not target_col or not rho_col:
        # If inference fails, still emit the original table in bundled form.
        out = df.copy()
    else:
        out = pd.DataFrame(
            {
                "Model": df[model_col],
                "Target": df[target_col],
                "NLL correlation ρ": df[rho_col].map(fmt_signed),
            }
        )
        if p_col:
            out["p"] = df[p_col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")

    return write_table_bundle(
        out,
        "table_4_item_nll_correlations",
        "Item-level unconditional language-model predictability correlations with human target ratings.",
    )



def make_table_5_absolute_effect_uncertainty() -> list[Path]:
    require(ABSOLUTE_EFFECTS)
    df = pd.read_csv(ABSOLUTE_EFFECTS)

    needed = {
        "method",
        "feature_set",
        "metric",
        "observed_mean_rho",
        "ci95_low",
        "ci95_high",
        "n_boot",
        "ci_excludes_zero",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"absolute effect table missing columns: {sorted(missing)}")

    feature_order = {
        "other_human_targets": 0,
        "other_human_plus_surface": 1,
        "stacked": 2,
    }
    method_order = {
        "compression_distilgpt2": 0,
        "embedding_contrast": 1,
        "tfidf_contrast": 2,
    }
    method_labels = {
        "compression_distilgpt2": "Compression",
        "embedding_contrast": "Embedding",
        "tfidf_contrast": "TF-IDF",
    }

    sub = df[
        (df["metric"] == "score_pref_struct")
        & (df["feature_set"].isin(feature_order))
        & (df["method"].isin(method_order))
    ].copy()

    sub["feature_order"] = sub["feature_set"].map(feature_order)
    sub["method_order"] = sub["method"].map(method_order)
    sub = sub.sort_values(["feature_order", "method_order"])

    out = pd.DataFrame(
        {
            "Readout": sub["method"].map(method_labels).fillna(sub["method"]),
            "Controls": sub["feature_set"].map(FEATURE_SET_LABELS).fillna(sub["feature_set"]),
            "Observed ρ": sub["observed_mean_rho"].map(fmt_signed),
            "95% CI": [
                f"[{fmt_signed(lo)}, {fmt_signed(hi)}]"
                for lo, hi in zip(sub["ci95_low"], sub["ci95_high"])
            ],
            "Bootstrap n": sub["n_boot"],
            "Interpretation": [
                "Positive point estimate; CI excludes zero."
                if bool(excludes)
                else "Positive point estimate; CI includes zero."
                for excludes in sub["ci_excludes_zero"]
            ],
        }
    )

    return write_table_bundle(
        out,
        "table_5_absolute_effect_uncertainty",
        "Poem-level bootstrap uncertainty for absolute supervised Surprise effects.",
    )



def make_table_6_domain_contrast_bootstrap() -> list[Path]:
    require(DOMAIN_CONTRAST_SURPRISE)
    require(GENERIC_ABSOLUTE_SURPRISE)

    contrast = pd.read_csv(DOMAIN_CONTRAST_SURPRISE)
    generic = pd.read_csv(GENERIC_ABSOLUTE_SURPRISE)

    needed_contrast = {
        "domain_variant",
        "control_variant",
        "observed_mean_supervised_rho",
        "observed_mean_diff",
        "ci95_low",
        "ci95_high",
        "n_boot",
        "ci_excludes_zero",
    }
    needed_generic = {
        "domain_variant",
        "control_variant",
        "observed_mean_rho",
        "ci95_low",
        "ci95_high",
    }
    missing_contrast = needed_contrast - set(contrast.columns)
    missing_generic = needed_generic - set(generic.columns)
    if missing_contrast:
        raise ValueError(f"domain contrast table missing columns: {sorted(missing_contrast)}")
    if missing_generic:
        raise ValueError(f"generic absolute table missing columns: {sorted(missing_generic)}")

    merged = contrast.merge(
        generic[
            [
                "domain_variant",
                "control_variant",
                "observed_mean_rho",
                "ci95_low",
                "ci95_high",
            ]
        ],
        on=["domain_variant", "control_variant"],
        how="left",
        suffixes=("", "_generic"),
    )

    domain_labels = {
        "accessible": "Accessible Gutenberg",
        "formal": "Formal Gutenberg",
    }
    control_labels = {
        "matchedctrl": "Matched-other",
        "wordctrl": "Word-shuffle",
    }
    order_domain = {"accessible": 0, "formal": 1}
    order_control = {"matchedctrl": 0, "wordctrl": 1}
    merged["domain_order"] = merged["domain_variant"].map(order_domain).fillna(99)
    merged["control_order"] = merged["control_variant"].map(order_control).fillna(99)
    merged = merged.sort_values(["domain_order", "control_order"])

    out = pd.DataFrame(
        {
            "Generic domain": merged["domain_variant"].map(domain_labels).fillna(merged["domain_variant"]),
            "Control": merged["control_variant"].map(control_labels).fillna(merged["control_variant"]),
            "Generic ρ": merged["observed_mean_rho"].map(fmt_signed),
            "Generic 95% CI": [
                f"[{fmt_signed(lo)}, {fmt_signed(hi)}]"
                for lo, hi in zip(merged["ci95_low_generic"], merged["ci95_high_generic"])
            ],
            "Supervised ρ": merged["observed_mean_supervised_rho"].map(fmt_signed),
            "Δρ supervised − generic": merged["observed_mean_diff"].map(fmt_signed),
            "Δρ 95% CI": [
                f"[{fmt_signed(lo)}, {fmt_signed(hi)}]"
                for lo, hi in zip(merged["ci95_low"], merged["ci95_high"])
            ],
            "Bootstrap n": merged["n_boot"],
            "Interpretation": [
                "Resolved paired improvement."
                if bool(excludes)
                else "Positive point estimate; CI includes zero."
                for excludes in merged["ci_excludes_zero"]
            ],
        }
    )

    return write_table_bundle(
        out,
        "table_6_domain_contrast_bootstrap",
        "Paired poem-level bootstrap comparing Surprise-shaped domains against generic Gutenberg domains.",
    )



DOMAIN_CONTRAST_SUMMARY_FILES = {
    "surprise_self": "bootstrap_domain_contrast_summary.csv",
    "aesthetic_self": "bootstrap_domain_contrast_summary_aesthetic_self.csv",
    "creativity_self": "bootstrap_domain_contrast_summary_creativity_self.csv",
    "pool_surprise_eval_aesthetic": "bootstrap_domain_contrast_summary_pool_surprise_eval_aesthetic.csv",
    "pool_aesthetic_eval_surprise": "bootstrap_domain_contrast_summary_pool_aesthetic_eval_surprise.csv",
    "pool_aesthetic_eval_creativity": "bootstrap_domain_contrast_summary_pool_aesthetic_eval_creativity.csv",
    "pool_creativity_eval_aesthetic": "bootstrap_domain_contrast_summary_pool_creativity_eval_aesthetic.csv",
    "pool_surprise_eval_creativity": "bootstrap_domain_contrast_summary_pool_surprise_eval_creativity.csv",
    "pool_creativity_eval_surprise": "bootstrap_domain_contrast_summary_pool_creativity_eval_surprise.csv",
}


def load_domain_contrast_all_controls() -> pd.DataFrame:
    frames = []
    for run_tag, filename in DOMAIN_CONTRAST_SUMMARY_FILES.items():
        source = ANALYSES_DIR / filename
        require(source)
        df = pd.read_csv(source)
        df["run_tag"] = run_tag
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    needed = {
        "pool_target",
        "target",
        "domain_variant",
        "control_variant",
        "observed_mean_supervised_rho",
        "observed_mean_generic_rho",
        "observed_mean_diff",
        "ci95_low",
        "ci95_high",
        "ci_excludes_zero",
    }
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"domain contrast summaries missing columns: {sorted(missing)}")
    return df


def make_table_7_domain_contrast_target_pair_matrix() -> list[Path]:
    df = load_domain_contrast_all_controls()

    counts = (
        df.groupby(["pool_target", "target"], dropna=False)
        .agg(
            n_controls=("control_variant", "count"),
            n_resolved=("ci_excludes_zero", "sum"),
            mean_delta=("observed_mean_diff", "mean"),
            min_ci_low=("ci95_low", "min"),
            max_ci_high=("ci95_high", "max"),
            supervised_rho=("observed_mean_supervised_rho", "mean"),
        )
        .reset_index()
        .sort_values(["pool_target", "target"])
    )

    out = pd.DataFrame(
        {
            "Pool": counts["pool_target"],
            "Eval": counts["target"],
            "Supervised ρ": counts["supervised_rho"].map(fmt_signed),
            "Mean Δρ": counts["mean_delta"].map(fmt_signed),
            "Resolved": counts["n_resolved"].astype(int).astype(str) + "/" + counts["n_controls"].astype(int).astype(str),
            "CI range": [
                f"[{fmt_signed(lo)}, {fmt_signed(hi)}]"
                for lo, hi in zip(counts["min_ci_low"], counts["max_ci_high"])
            ],
        }
    )

    return write_table_bundle(
        out,
        "table_7_domain_contrast_target_pair_matrix",
        "Consolidated paired domain-contrast matrix across pool and evaluation targets.",
    )


def make_table_8_domain_contrast_by_control_family() -> list[Path]:
    df = load_domain_contrast_all_controls()

    df["control_family"] = df["control_variant"].map(
        lambda x: "word-shuffle" if str(x) == "wordctrl" else "matched-control"
    )

    summary = (
        df.groupby(["pool_target", "target", "control_family"], dropna=False)
        .agg(
            n_controls=("control_variant", "count"),
            n_resolved=("ci_excludes_zero", "sum"),
            mean_delta=("observed_mean_diff", "mean"),
            min_ci_low=("ci95_low", "min"),
            max_ci_high=("ci95_high", "max"),
            mean_supervised_rho=("observed_mean_supervised_rho", "mean"),
            mean_generic_rho=("observed_mean_generic_rho", "mean"),
        )
        .reset_index()
        .sort_values(["pool_target", "target", "control_family"])
    )

    out = pd.DataFrame(
        {
            "Pool": summary["pool_target"],
            "Eval": summary["target"],
            "Control": summary["control_family"],
            "Supervised ρ": summary["mean_supervised_rho"].map(fmt_signed),
            "Generic ρ": summary["mean_generic_rho"].map(fmt_signed),
            "Mean Δρ": summary["mean_delta"].map(fmt_signed),
            "Resolved": summary["n_resolved"].astype(int).astype(str) + "/" + summary["n_controls"].astype(int).astype(str),
            "CI range": [
                f"[{fmt_signed(lo)}, {fmt_signed(hi)}]"
                for lo, hi in zip(summary["min_ci_low"], summary["max_ci_high"])
            ],
        }
    )

    return write_table_bundle(
        out,
        "table_8_domain_contrast_by_control_family",
        "Domain-contrast results split by matched-control and word-shuffle control families.",
    )


def hash_entry(path: Path, note: str) -> dict:
    rows, cols = csv_shape(path)
    return {
        "status": "generated",
        "destination": str(path),
        "sha256": sha256_file(path),
        "n_rows": rows,
        "n_cols": cols,
        "note": note,
    }


def write_hash_csv(entries: list[dict]) -> None:
    HASH_DIR.mkdir(parents=True, exist_ok=True)
    out = HASH_DIR / "table_hashes.csv"

    fieldnames = [
        "status",
        "destination",
        "sha256",
        "n_rows",
        "n_cols",
        "note",
    ]

    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            writer.writerow({k: e.get(k, "") for k in fieldnames})


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    HASH_DIR.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    generated.extend(make_table_1_readout_convergence())
    generated.extend(make_table_2_bootstrap_uncertainty())
    generated.extend(make_table_3_generic_d_summary())
    generated.extend(make_table_4_item_nll_correlations())
    generated.extend(make_table_5_absolute_effect_uncertainty())
    generated.extend(make_table_6_domain_contrast_bootstrap())
    generated.extend(make_table_7_domain_contrast_target_pair_matrix())
    generated.extend(make_table_8_domain_contrast_by_control_family())

    manifest = {
        "stage": "90_make_tables",
        "outputs": [str(p) for p in generated],
        "important_note": (
            "This stage formats existing analysis outputs into paper-facing tables. "
            "It does not change or recompute empirical values."
        ),
    }

    manifest_path = TABLE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    entries = [
        hash_entry(p, "Generated paper table artifact.")
        for p in generated
    ]
    write_hash_csv(entries)

    print("Generated paper tables:")
    for p in generated:
        rows, cols = csv_shape(p)
        shape = f" rows={rows} cols={cols}" if p.suffix == ".csv" else ""
        print(f"  {p}{shape}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {HASH_DIR / 'table_hashes.csv'}")


if __name__ == "__main__":
    main()
