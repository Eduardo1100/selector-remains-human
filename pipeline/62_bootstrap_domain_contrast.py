#!/usr/bin/env python3
"""
Bootstrap generic-D absolute effects and paired generic-vs-supervised domain contrasts.

This script addresses the critique that the generic negative claim and supervised positive
claim must be evaluated under the same poem-level uncertainty standard.

Main tests:
  1. Generic absolute:
       rho(generic_score, target)
  2. Paired domain contrast:
       rho(supervised_pref_score, target) - rho(generic_score, target)

The paired contrast is computed on the same Chaudhuri poems and same target, so poem-level
variance is shared across both correlations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "results" / "scores" / "phase_a_eval_scores"
DATA = ROOT / "data" / "processed"
ANALYSES = ROOT / "results" / "analyses"
HASHES = ROOT / "results" / "hashes"


GENERIC_PATTERN = re.compile(
    r"vscore_distilgpt2_gutenberg_(?P<domain_variant>accessible|formal)_v0_"
    r"(?P<control_variant>matchedctrl|wordctrl)_seed(?P<seed>\d+)_dn(?P<d_n>\d+)\.csv$"
)

SUPERVISED_PATTERN = re.compile(
    r"vscore_distilgpt2_kfold_surface_pool(?P<pool_target>.+?)_eval(?P<eval_target>.+?)_"
    r"foldseed(?P<fold_seed>\d+)_seed(?P<seed>\d+)_dn(?P<d_n>\d+)\.csv$"
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_spearman(x: pd.Series, y: pd.Series) -> float:
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan
    if x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(spearmanr(x[ok], y[ok]).correlation)


def load_targets() -> pd.DataFrame:
    targets = pd.read_csv(DATA / "targets_wide.csv")
    if "item_id" not in targets.columns:
        raise ValueError("targets_wide.csv must contain item_id")
    return targets


def target_col_name(target: str) -> str:
    c = f"target__{target}"
    return c


def load_generic_scores(target: str, metric: str) -> pd.DataFrame:
    targets = load_targets()
    tcol = target_col_name(target)
    if tcol not in targets.columns:
        raise ValueError(f"Missing target column {tcol} in targets_wide.csv")

    rows = []
    for path in sorted(SCORES.glob("vscore_distilgpt2_gutenberg_*_v0_*_seed*_dn32.csv")):
        m = GENERIC_PATTERN.match(path.name)
        if not m:
            continue

        meta = m.groupdict()
        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        if "item_id" not in df.columns:
            continue

        # Restrict to Chaudhuri, because the supervised K-fold scores are Chaudhuri-only.
        if "dataset" in df.columns:
            df = df[df["dataset"] == "chaudhuri_2024"].copy()
        else:
            df = df[df["item_id"].astype(str).str.startswith("chaudhuri")].copy()

        df = df[["item_id", metric]].rename(columns={metric: "generic_score"})
        df = df.merge(targets[["item_id", tcol]], on="item_id", how="inner")
        df = df.rename(columns={tcol: "target_value"})

        for k, v in meta.items():
            df[k] = v
        df["source_file"] = path.name
        df["generic_metric"] = metric
        df["target"] = target
        rows.append(df)

    if not rows:
        raise RuntimeError("No generic score rows found.")

    out = pd.concat(rows, ignore_index=True)
    return out


def load_supervised_scores(target: str, metric: str, pool_target: str | None = None) -> pd.DataFrame:
    targets = load_targets()
    tcol = target_col_name(target)
    if tcol not in targets.columns:
        raise ValueError(f"Missing target column {tcol} in targets_wide.csv")

    rows = []
    for path in sorted(SCORES.glob("vscore_distilgpt2_kfold_surface_pool*_eval*_foldseed*_seed*_dn8.csv")):
        m = SUPERVISED_PATTERN.match(path.name)
        if not m:
            continue

        meta = m.groupdict()
        if meta["eval_target"] != target:
            continue
        if pool_target is not None and meta["pool_target"] != pool_target:
            continue

        df = pd.read_csv(path)
        if metric not in df.columns:
            continue
        if "item_id" not in df.columns:
            continue

        # Supervised K-fold files are Chaudhuri-only, but keep this explicit.
        if "dataset" in df.columns:
            df = df[df["dataset"] == "chaudhuri_2024"].copy()
        else:
            df = df[df["item_id"].astype(str).str.startswith("chaudhuri")].copy()

        df = df[["item_id", metric]].rename(columns={metric: "supervised_score"})
        df = df.merge(targets[["item_id", tcol]], on="item_id", how="inner")
        df = df.rename(columns={tcol: "target_value"})

        for k, v in meta.items():
            df[k] = v
        df["source_file"] = path.name
        df["supervised_metric"] = metric
        df["target"] = target
        rows.append(df)

    if not rows:
        raise RuntimeError("No supervised score rows found.")

    out = pd.concat(rows, ignore_index=True)
    return out


def summarize_bootstrap(samples: list[float]) -> dict:
    arr = np.array(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    return {
        "boot_mean": float(np.mean(arr)),
        "boot_std": float(np.std(arr, ddof=1)),
        "ci95_low": float(np.quantile(arr, 0.025)),
        "ci95_high": float(np.quantile(arr, 0.975)),
        "p_boot_le_zero": float(np.mean(arr <= 0)),
        "p_boot_ge_zero": float(np.mean(arr >= 0)),
        "n_boot": int(len(arr)),
        "ci_excludes_zero": bool((np.quantile(arr, 0.025) > 0) or (np.quantile(arr, 0.975) < 0)),
    }


def bootstrap_generic_absolute(generic: pd.DataFrame, n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap generic absolute effects using one shared poem resample per bootstrap ID.

    This avoids treating seeds / domain variants as independent item resamples. Each
    bootstrap replicate samples the poem IDs once, then evaluates every generic run
    on that same resampled item set.
    """
    rng = np.random.default_rng(seed)
    observed_rows = []
    sample_rows = []

    group_cols = ["target", "domain_variant", "control_variant", "seed", "d_n", "generic_metric", "source_file"]
    groups = list(generic.groupby(group_cols, dropna=False))

    item_sets = [set(df["item_id"].unique()) for _, df in groups]
    common_item_ids = sorted(set.intersection(*item_sets))
    if len(common_item_ids) < 10:
        raise RuntimeError(f"Too few common generic items: {len(common_item_ids)}")

    indexed_groups = []
    for key, df in groups:
        meta = dict(zip(group_cols, key))
        df = df[df["item_id"].isin(common_item_ids)].copy()
        indexed_groups.append((meta, df))

        observed = safe_spearman(df["generic_score"], df["target_value"])
        observed_rows.append({
            **meta,
            "analysis": "generic_absolute",
            "n_items": len(common_item_ids),
            "observed_rho": observed,
        })

    for b in range(n_boot):
        sampled = rng.choice(common_item_ids, size=len(common_item_ids), replace=True)
        sample_index = pd.DataFrame({
            "item_id": sampled,
            "boot_item_instance": np.arange(len(sampled)),
        })

        for meta, df in indexed_groups:
            boot = sample_index.merge(df, on="item_id", how="left")
            rho = safe_spearman(boot["generic_score"], boot["target_value"])
            sample_rows.append({
                **meta,
                "analysis": "generic_absolute",
                "boot_id": b,
                "boot_rho": rho,
            })

        if n_boot >= 1000 and (b + 1) % 500 == 0:
            print(f"  generic absolute bootstrap {b + 1}/{n_boot}")

    return pd.DataFrame(observed_rows), pd.DataFrame(sample_rows)


def bootstrap_domain_contrast(
    generic: pd.DataFrame,
    supervised: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Bootstrap paired generic-vs-supervised domain contrasts using one shared poem
    resample per bootstrap ID.

    This is the key paired test:
        rho(supervised_D_score, target) - rho(generic_D_score, target)

    The same resampled poem IDs are used for every generic/supervised run-pair
    inside each bootstrap replicate, so uncertainty remains item-level rather
    than run-level.
    """
    rng = np.random.default_rng(seed)
    observed_rows = []
    sample_rows = []

    generic_group_cols = ["target", "domain_variant", "control_variant", "seed", "d_n", "generic_metric", "source_file"]
    supervised_group_cols = ["target", "pool_target", "eval_target", "fold_seed", "seed", "d_n", "supervised_metric", "source_file"]

    generic_groups = []
    for key, df in generic.groupby(generic_group_cols, dropna=False):
        meta = dict(zip(generic_group_cols, key))
        generic_groups.append((meta, df.copy()))

    supervised_groups = []
    for key, df in supervised.groupby(supervised_group_cols, dropna=False):
        meta = dict(zip(supervised_group_cols, key))
        supervised_groups.append((meta, df.copy()))

    # Use the common item universe across all runs in the paired analysis.
    item_sets = []
    for _, df in generic_groups:
        item_sets.append(set(df["item_id"].unique()))
    for _, df in supervised_groups:
        item_sets.append(set(df["item_id"].unique()))

    common_item_ids = sorted(set.intersection(*item_sets))
    if len(common_item_ids) < 10:
        raise RuntimeError(f"Too few common contrast items: {len(common_item_ids)}")

    pair_rows = []
    for gmeta, gdf in generic_groups:
        for smeta, sdf in supervised_groups:
            if gmeta["target"] != smeta["target"]:
                continue

            merged = (
                gdf[["item_id", "target_value", "generic_score"]]
                .merge(
                    sdf[["item_id", "supervised_score"]],
                    on="item_id",
                    how="inner",
                )
            )
            merged = merged[merged["item_id"].isin(common_item_ids)].copy()
            if merged["item_id"].nunique() != len(common_item_ids):
                continue

            rho_generic = safe_spearman(merged["generic_score"], merged["target_value"])
            rho_supervised = safe_spearman(merged["supervised_score"], merged["target_value"])
            diff = rho_supervised - rho_generic

            row_meta = {
                "analysis": "domain_contrast",
                "target": gmeta["target"],
                "domain_variant": gmeta["domain_variant"],
                "control_variant": gmeta["control_variant"],
                "generic_seed": gmeta["seed"],
                "generic_d_n": gmeta["d_n"],
                "generic_metric": gmeta["generic_metric"],
                "generic_source_file": gmeta["source_file"],
                "pool_target": smeta["pool_target"],
                "eval_target": smeta["eval_target"],
                "fold_seed": smeta["fold_seed"],
                "supervised_seed": smeta["seed"],
                "supervised_d_n": smeta["d_n"],
                "supervised_metric": smeta["supervised_metric"],
                "supervised_source_file": smeta["source_file"],
                "n_items": len(common_item_ids),
            }

            observed_rows.append({
                **row_meta,
                "observed_generic_rho": rho_generic,
                "observed_supervised_rho": rho_supervised,
                "observed_diff": diff,
            })

            pair_rows.append((row_meta, merged))

    for b in range(n_boot):
        sampled = rng.choice(common_item_ids, size=len(common_item_ids), replace=True)
        sample_index = pd.DataFrame({
            "item_id": sampled,
            "boot_item_instance": np.arange(len(sampled)),
        })

        for row_meta, merged in pair_rows:
            boot = sample_index.merge(merged, on="item_id", how="left")
            boot_g = safe_spearman(boot["generic_score"], boot["target_value"])
            boot_s = safe_spearman(boot["supervised_score"], boot["target_value"])
            sample_rows.append({
                **row_meta,
                "boot_id": b,
                "boot_generic_rho": boot_g,
                "boot_supervised_rho": boot_s,
                "boot_diff": boot_s - boot_g,
            })

        if n_boot >= 1000 and (b + 1) % 500 == 0:
            print(f"  domain contrast bootstrap {b + 1}/{n_boot}")

    return pd.DataFrame(observed_rows), pd.DataFrame(sample_rows)


def make_summaries(
    generic_obs: pd.DataFrame,
    generic_samples: pd.DataFrame,
    contrast_obs: pd.DataFrame,
    contrast_samples: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gen_group_cols = ["target", "domain_variant", "control_variant", "generic_metric"]
    gen_summary_rows = []

    for key, obs_group in generic_obs.groupby(gen_group_cols, dropna=False):
        meta = dict(zip(gen_group_cols, key))
        sample_group = generic_samples
        for k, v in meta.items():
            sample_group = sample_group[sample_group[k] == v]

        vals = []
        # Average across generic seeds inside each bootstrap ID, matching the run-level summary logic.
        for boot_id, bdf in sample_group.groupby("boot_id"):
            vals.append(float(bdf["boot_rho"].mean()))

        obs_mean = float(obs_group["observed_rho"].mean())
        gen_summary_rows.append({
            **meta,
            "analysis": "generic_absolute_summary",
            "n_runs": int(len(obs_group)),
            "observed_mean_rho": obs_mean,
            **summarize_bootstrap(vals),
        })

    contrast_group_cols = [
        "target",
        "domain_variant",
        "control_variant",
        "generic_metric",
        "pool_target",
        "supervised_metric",
    ]
    contrast_summary_rows = []

    for key, obs_group in contrast_obs.groupby(contrast_group_cols, dropna=False):
        meta = dict(zip(contrast_group_cols, key))
        sample_group = contrast_samples
        for k, v in meta.items():
            sample_group = sample_group[sample_group[k] == v]

        vals = []
        # Average across all generic seeds, fold seeds, and supervised seeds per bootstrap ID.
        for boot_id, bdf in sample_group.groupby("boot_id"):
            vals.append(float(bdf["boot_diff"].mean()))

        contrast_summary_rows.append({
            **meta,
            "analysis": "domain_contrast_summary",
            "n_pairs": int(len(obs_group)),
            "observed_mean_generic_rho": float(obs_group["observed_generic_rho"].mean()),
            "observed_mean_supervised_rho": float(obs_group["observed_supervised_rho"].mean()),
            "observed_mean_diff": float(obs_group["observed_diff"].mean()),
            **summarize_bootstrap(vals),
        })

    return pd.DataFrame(gen_summary_rows), pd.DataFrame(contrast_summary_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="Surprise")
    ap.add_argument("--pool-target", default="Surprise")
    ap.add_argument("--generic-metric", default="v_struct")
    ap.add_argument("--supervised-metric", default="v_pref_struct")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tag", default="", help="Optional output tag to avoid overwriting prior runs.")
    args = ap.parse_args()

    def tagged(name: str) -> str:
        if not args.tag:
            return name
        stem, suffix = name.rsplit(".", 1)
        return f"{stem}_{args.tag}.{suffix}"

    ANALYSES.mkdir(parents=True, exist_ok=True)
    HASHES.mkdir(parents=True, exist_ok=True)

    print(f"Target: {args.target}")
    print(f"Pool target: {args.pool_target}")
    print(f"Generic metric: {args.generic_metric}")
    print(f"Supervised metric: {args.supervised_metric}")
    print(f"Bootstraps: {args.n_boot}")

    generic = load_generic_scores(target=args.target, metric=args.generic_metric)
    supervised = load_supervised_scores(
        target=args.target,
        metric=args.supervised_metric,
        pool_target=args.pool_target,
    )

    print(f"Generic rows: {len(generic)}")
    print(f"Supervised rows: {len(supervised)}")
    print(f"Generic items: {generic['item_id'].nunique()}")
    print(f"Supervised items: {supervised['item_id'].nunique()}")

    generic_obs, generic_samples = bootstrap_generic_absolute(generic, args.n_boot, args.seed)
    contrast_obs, contrast_samples = bootstrap_domain_contrast(generic, supervised, args.n_boot, args.seed + 1)
    generic_summary, contrast_summary = make_summaries(
        generic_obs,
        generic_samples,
        contrast_obs,
        contrast_samples,
    )

    outputs = {
        tagged("bootstrap_generic_absolute_observed.csv"): generic_obs,
        tagged("bootstrap_generic_absolute_samples.csv"): generic_samples,
        tagged("bootstrap_generic_absolute_summary.csv"): generic_summary,
        tagged("bootstrap_domain_contrast_observed.csv"): contrast_obs,
        tagged("bootstrap_domain_contrast_samples.csv"): contrast_samples,
        tagged("bootstrap_domain_contrast_summary.csv"): contrast_summary,
    }

    for name, df in outputs.items():
        path = ANALYSES / name
        df.to_csv(path, index=False)
        print(f"Wrote {path} rows={len(df)} cols={len(df.columns)}")

    manifest = {
        "script": Path(__file__).name,
        "target": args.target,
        "pool_target": args.pool_target,
        "generic_metric": args.generic_metric,
        "supervised_metric": args.supervised_metric,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "outputs": sorted(outputs),
    }
    manifest_path = ANALYSES / tagged("bootstrap_domain_contrast_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")

    hash_rows = []
    for name in list(outputs) + [manifest_path.name]:
        path = ANALYSES / name
        hash_rows.append({
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    hash_df = pd.DataFrame(hash_rows)
    hash_path = HASHES / tagged("bootstrap_domain_contrast_hashes.csv")
    hash_df.to_csv(hash_path, index=False)
    print(f"Wrote {hash_path}")

    print("\nGeneric absolute summary:")
    print(generic_summary.to_string(index=False))

    print("\nDomain contrast summary:")
    print(contrast_summary.to_string(index=False))


if __name__ == "__main__":
    main()
