from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCORE_DIR = ROOT / "results" / "scores" / "phase_a_eval_scores"
ANALYSES_DIR = ROOT / "results" / "analyses"
HASH_DIR = ROOT / "results" / "hashes"

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

def spearman_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(spearmanr(x, y, nan_policy="omit").statistic)

def find_prefcontrast_files(observer: str, label: str) -> dict[str, list[Path]]:
    files = sorted(
        p for p in SCORE_DIR.glob(f"vscore_{observer}_prefcontrast_chaudhuri_{label}_*.csv")
        if not p.name.endswith(".correlations.csv")
    )
    out: dict[str, list[Path]] = {}
    for p in files:
        rest = p.name.split(f"chaudhuri_{label}_", 1)[1]
        control = rest.split("_", 1)[0]
        out.setdefault(control, []).append(p)
    return out

def load_item_mean(paths: list[Path], metric: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        required = {"item_id", "target_value", metric}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{p} missing columns: {sorted(missing)}")

        keep_cols = ["item_id", "target_value", metric]
        if "title" in df.columns:
            keep_cols.append("title")
        if "poet" in df.columns:
            keep_cols.append("poet")

        sub = df[keep_cols].copy()
        sub["source_file"] = str(p)
        frames.append(sub)

    all_df = pd.concat(frames, ignore_index=True)

    # target_value should be identical per item across runs.
    grouped = (
        all_df.groupby("item_id", dropna=False)
        .agg(
            target_value=("target_value", "mean"),
            score=(metric, "mean"),
            score_std=(metric, "std"),
            n_runs=(metric, "count"),
        )
        .reset_index()
    )

    return grouped

def run_bootstrap(
    human: pd.DataFrame,
    null: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    merged = human.merge(
        null,
        on="item_id",
        how="inner",
        suffixes=("_human", "_null"),
    )

    if merged.empty:
        raise ValueError("No overlapping item_id values between human and null files.")

    # Check target values agree. Use human as canonical but flag if mismatch is non-trivial.
    max_target_diff = float(np.max(np.abs(merged["target_value_human"] - merged["target_value_null"])))

    y = merged["target_value_human"].to_numpy(float)
    human_score = merged["score_human"].to_numpy(float)
    null_score = merged["score_null"].to_numpy(float)

    obs_human = spearman_safe(human_score, y)
    obs_null = spearman_safe(null_score, y)
    obs_diff = obs_human - obs_null

    rng = np.random.default_rng(seed)
    rows = []
    n = len(merged)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rho_h = spearman_safe(human_score[idx], y[idx])
        rho_n = spearman_safe(null_score[idx], y[idx])
        rows.append(
            {
                "boot_id": b,
                "rho_human": rho_h,
                "rho_null": rho_n,
                "diff_human_minus_null": rho_h - rho_n,
            }
        )

    samples = pd.DataFrame(rows)
    diff = samples["diff_human_minus_null"].dropna().to_numpy(float)

    summary = {
        "n_items": n,
        "observed_human_rho": obs_human,
        "observed_null_rho": obs_null,
        "observed_diff": obs_diff,
        "boot_mean_diff": float(np.mean(diff)),
        "boot_std_diff": float(np.std(diff, ddof=1)),
        "ci95_low": float(np.quantile(diff, 0.025)),
        "ci95_high": float(np.quantile(diff, 0.975)),
        "p_boot_le_zero": float(np.mean(diff <= 0)),
        "p_boot_ge_zero": float(np.mean(diff >= 0)),
        "n_boot": n_boot,
        "ci_excludes_zero": bool(np.quantile(diff, 0.025) > 0 or np.quantile(diff, 0.975) < 0),
        "max_target_value_diff_between_inputs": max_target_diff,
    }
    return summary, samples

def write_hashes(paths: list[Path], tag: str) -> None:
    HASH_DIR.mkdir(parents=True, exist_ok=True)
    out = HASH_DIR / f"bootstrap_same_form_domain_contrast_hashes_{tag}.csv"
    rows = []
    for p in paths:
        r, c = csv_shape(p)
        rows.append(
            {
                "path": str(p),
                "sha256": sha256_file(p),
                "n_rows": r,
                "n_cols": c,
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--observer", default="distilgpt2")
    ap.add_argument("--human-label", default="Surprise")
    ap.add_argument("--null-label", default="Surprise_surfacepool")
    ap.add_argument("--metric", default="v_pref_struct")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tag", default="surprise_surfacepool")
    args = ap.parse_args()

    ANALYSES_DIR.mkdir(parents=True, exist_ok=True)
    HASH_DIR.mkdir(parents=True, exist_ok=True)

    human_files_by_control = find_prefcontrast_files(args.observer, args.human_label)
    null_files_by_control = find_prefcontrast_files(args.observer, args.null_label)

    common_controls = sorted(set(human_files_by_control) & set(null_files_by_control))
    if not common_controls:
        raise SystemExit(
            f"No common controls between {args.human_label} and {args.null_label}.\n"
            f"human={sorted(human_files_by_control)} null={sorted(null_files_by_control)}"
        )

    summary_rows = []
    observed_rows = []
    all_sample_frames = []
    input_files = []

    for control in common_controls:
        human_paths = human_files_by_control[control]
        null_paths = null_files_by_control[control]
        input_files.extend(human_paths)
        input_files.extend(null_paths)

        human = load_item_mean(human_paths, args.metric)
        null = load_item_mean(null_paths, args.metric)

        summary, samples = run_bootstrap(human, null, n_boot=args.n_boot, seed=args.seed)
        summary.update(
            {
                "observer": args.observer,
                "human_label": args.human_label,
                "null_label": args.null_label,
                "control": control,
                "metric": args.metric,
                "n_human_files": len(human_paths),
                "n_null_files": len(null_paths),
                "analysis": "same_form_prefcontrast_item_bootstrap",
            }
        )
        summary_rows.append(summary)

        observed_rows.append(
            {
                "observer": args.observer,
                "human_label": args.human_label,
                "null_label": args.null_label,
                "control": control,
                "metric": args.metric,
                "n_items": summary["n_items"],
                "observed_human_rho": summary["observed_human_rho"],
                "observed_null_rho": summary["observed_null_rho"],
                "observed_diff": summary["observed_diff"],
                "n_human_files": len(human_paths),
                "n_null_files": len(null_paths),
            }
        )

        samples["observer"] = args.observer
        samples["human_label"] = args.human_label
        samples["null_label"] = args.null_label
        samples["control"] = control
        samples["metric"] = args.metric
        all_sample_frames.append(samples)

        print(
            f"{control}: human_rho={summary['observed_human_rho']:+.3f} "
            f"null_rho={summary['observed_null_rho']:+.3f} "
            f"diff={summary['observed_diff']:+.3f} "
            f"CI=[{summary['ci95_low']:+.3f}, {summary['ci95_high']:+.3f}] "
            f"resolved={summary['ci_excludes_zero']}"
        )

    observed = pd.DataFrame(observed_rows)
    summary_df = pd.DataFrame(summary_rows)
    samples_df = pd.concat(all_sample_frames, ignore_index=True)

    observed_path = ANALYSES_DIR / f"bootstrap_same_form_domain_contrast_observed_{args.tag}.csv"
    summary_path = ANALYSES_DIR / f"bootstrap_same_form_domain_contrast_summary_{args.tag}.csv"
    samples_path = ANALYSES_DIR / f"bootstrap_same_form_domain_contrast_samples_{args.tag}.csv"
    manifest_path = ANALYSES_DIR / f"bootstrap_same_form_domain_contrast_manifest_{args.tag}.json"

    observed.to_csv(observed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    samples_df.to_csv(samples_path, index=False)

    manifest = {
        "analysis": "same_form_prefcontrast_item_bootstrap",
        "observer": args.observer,
        "human_label": args.human_label,
        "null_label": args.null_label,
        "metric": args.metric,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "tag": args.tag,
        "common_controls": common_controls,
        "inputs": [str(p) for p in sorted(set(input_files))],
        "outputs": [str(observed_path), str(summary_path), str(samples_path), str(manifest_path)],
        "interpretation_note": (
            "Both sides use the same prefcontrast metric. This tests human-labeled contrastive "
            "domains against a nonhuman contrastive domain artifact, avoiding the previous "
            "v_pref_struct-vs-v_struct mixed-form contrast."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_hashes([observed_path, summary_path, manifest_path, *sorted(set(input_files))], args.tag)

    print(f"Wrote {observed_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {samples_path}")
    print(f"Wrote {manifest_path}")

if __name__ == "__main__":
    main()
