#!/usr/bin/env python3
"""
Bootstrap observer-family robustness for existing prefcontrast score files.

This is a bounded observer-family check:
- It does not require new LM scoring.
- It uses existing vscore_{observer}_prefcontrast_chaudhuri_{target}_matchedctrl_frac33_seed*_dn8.csv files.
- It estimates item-level uncertainty for v_pref_struct alignment with the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "scores" / "phase_a_eval_scores"
ANALYSES = ROOT / "results" / "analyses"
HASHES = ROOT / "results" / "hashes"


def safe_spearman(x, y) -> float:
    x = pd.Series(x)
    y = pd.Series(y)
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return float("nan")
    if x[mask].nunique() < 2 or y[mask].nunique() < 2:
        return float("nan")
    return float(spearmanr(x[mask], y[mask]).statistic)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scores(observer: str, target: str, metric: str) -> pd.DataFrame:
    pattern = f"vscore_{observer}_prefcontrast_chaudhuri_{target}_matchedctrl_frac33_seed*_dn8.csv"
    paths = sorted(p for p in SCORES.glob(pattern) if not p.name.endswith(".correlations.csv"))
    if not paths:
        raise FileNotFoundError(f"No files matched {pattern}")

    rows = []
    for path in paths:
        df = pd.read_csv(path)
        if metric not in df.columns:
            raise ValueError(f"{path} missing metric column {metric}")

        # Seed is present as a column in these files, but keep source-file metadata too.
        keep = df[["item_id", "target_value", metric, "observer", "seed", "d_n", "control_mode"]].copy()
        keep = keep.rename(columns={metric: "score"})
        keep["source_file"] = path.name
        keep["target"] = target
        rows.append(keep)

    out = pd.concat(rows, ignore_index=True)
    return out


def bootstrap(df: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    group_cols = ["observer", "target", "seed", "d_n", "control_mode", "source_file"]
    groups = list(df.groupby(group_cols, dropna=False))

    common_items = sorted(set.intersection(*(set(g["item_id"].unique()) for _, g in groups)))
    if len(common_items) < 10:
        raise RuntimeError(f"Too few common items: {len(common_items)}")

    observed_rows = []
    sample_rows = []

    indexed = []
    for key, g in groups:
        meta = dict(zip(group_cols, key))
        g = g[g["item_id"].isin(common_items)].copy()
        observed_rows.append({
            **meta,
            "analysis": "observer_family_absolute",
            "n_items": len(common_items),
            "observed_rho": safe_spearman(g["score"], g["target_value"]),
        })
        indexed.append((meta, g))

    for b in range(n_boot):
        sampled = rng.choice(common_items, size=len(common_items), replace=True)
        sample_index = pd.DataFrame({
            "item_id": sampled,
            "boot_item_instance": np.arange(len(sampled)),
        })

        for meta, g in indexed:
            boot = sample_index.merge(g, on="item_id", how="left")
            sample_rows.append({
                **meta,
                "analysis": "observer_family_absolute",
                "boot_id": b,
                "boot_rho": safe_spearman(boot["score"], boot["target_value"]),
            })

        if n_boot >= 1000 and (b + 1) % 500 == 0:
            print(f"  bootstrap {b + 1}/{n_boot}")

    return pd.DataFrame(observed_rows), pd.DataFrame(sample_rows)


def summarize(observed: pd.DataFrame, samples: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["observer", "target", "control_mode"]

    obs_summary = (
        observed
        .groupby(group_cols, dropna=False)
        .agg(
            n_runs=("observed_rho", "count"),
            observed_mean_rho=("observed_rho", "mean"),
        )
        .reset_index()
    )

    boot_by_id = (
        samples
        .groupby(group_cols + ["boot_id"], dropna=False)
        .agg(boot_mean_rho=("boot_rho", "mean"))
        .reset_index()
    )

    rows = []
    for key, g in boot_by_id.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, key))
        vals = g["boot_mean_rho"].dropna().to_numpy()
        rows.append({
            **meta,
            "analysis": "observer_family_absolute_summary",
            "boot_mean": float(np.mean(vals)),
            "boot_std": float(np.std(vals, ddof=1)),
            "ci95_low": float(np.quantile(vals, 0.025)),
            "ci95_high": float(np.quantile(vals, 0.975)),
            "p_boot_le_zero": float(np.mean(vals <= 0)),
            "p_boot_ge_zero": float(np.mean(vals >= 0)),
            "n_boot": int(len(vals)),
            "ci_excludes_zero": bool((np.quantile(vals, 0.025) > 0) or (np.quantile(vals, 0.975) < 0)),
        })

    summ = pd.DataFrame(rows)
    return obs_summary.merge(summ, on=group_cols, how="left")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Aesthetic_Appeal")
    ap.add_argument("--metric", default="v_pref_struct")
    ap.add_argument("--observers", nargs="+", default=["gpt2", "gpt2-medium"])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    tag = args.tag or f"{args.target}_observer_family"
    ANALYSES.mkdir(parents=True, exist_ok=True)
    HASHES.mkdir(parents=True, exist_ok=True)

    frames = []
    for obs in args.observers:
        print(f"Loading observer={obs} target={args.target}")
        frames.append(load_scores(obs, args.target, args.metric))

    df = pd.concat(frames, ignore_index=True)
    observed, samples = bootstrap(df, n_boot=args.n_boot, seed=args.seed)
    summary = summarize(observed, samples)

    outputs = {
        f"bootstrap_observer_family_observed_{tag}.csv": observed,
        f"bootstrap_observer_family_samples_{tag}.csv": samples,
        f"bootstrap_observer_family_summary_{tag}.csv": summary,
    }

    for name, out in outputs.items():
        path = ANALYSES / name
        out.to_csv(path, index=False)
        print(f"Wrote {path} rows={len(out)} cols={len(out.columns)}")

    manifest = {
        "target": args.target,
        "metric": args.metric,
        "observers": args.observers,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "tag": tag,
        "input_rows": int(len(df)),
        "input_files": sorted(df["source_file"].unique().tolist()),
        "outputs": sorted(outputs),
    }
    manifest_path = ANALYSES / f"bootstrap_observer_family_manifest_{tag}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")

    hash_rows = []
    for name in list(outputs) + [manifest_path.name]:
        path = ANALYSES / name
        hash_rows.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    hash_path = HASHES / f"bootstrap_observer_family_hashes_{tag}.csv"
    pd.DataFrame(hash_rows).to_csv(hash_path, index=False)
    print(f"Wrote {hash_path}")

    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
