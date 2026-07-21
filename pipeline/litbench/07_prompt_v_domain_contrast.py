from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"


def bootstrap_accuracy(correct: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    correct = np.asarray(correct, dtype=float)
    n = len(correct)

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(correct[idx]))

    return {
        "accuracy": float(np.mean(correct)),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "n": int(n),
        "n_boot": int(n_boot),
    }


def bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)

    boots = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = float(np.mean(values[idx]))

    return {
        "mean": float(np.mean(values)),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "p_le_zero": float(np.mean(boots <= 0.0)),
        "p_ge_zero": float(np.mean(boots >= 0.0)),
        "n": int(n),
        "n_boot": int(n_boot),
    }


def paired_bootstrap_delta(correct_a: np.ndarray, correct_b: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)

    if len(a) != len(b):
        raise ValueError("Paired arrays must have same length.")

    diff = a - b
    n = len(diff)

    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(diff[idx]))

    return {
        "delta_accuracy": float(np.mean(diff)),
        "delta_ci95_low": float(np.quantile(boots, 0.025)),
        "delta_ci95_high": float(np.quantile(boots, 0.975)),
        "p_delta_le_zero": float(np.mean(boots <= 0.0)),
        "p_delta_ge_zero": float(np.mean(boots >= 0.0)),
    }


def crossval_predictions(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)

    for train_idx, test_idx in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        )
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])

    return preds


def load_scores(chosen_tag: str, rejected_tag: str) -> pd.DataFrame:
    chosen_path = OUT_DIR / f"litbench_prompt_conditioned_v_scores_{chosen_tag}.csv"
    rejected_path = OUT_DIR / f"litbench_prompt_conditioned_v_scores_{rejected_tag}.csv"

    if not chosen_path.exists():
        raise FileNotFoundError(chosen_path)
    if not rejected_path.exists():
        raise FileNotFoundError(rejected_path)

    ch = pd.read_csv(chosen_path)
    rj = pd.read_csv(rejected_path)

    ch = ch.rename(
        columns={
            "chosen_gain": "chosen_gain_chosen_domain",
            "rejected_gain": "rejected_gain_chosen_domain",
            "v_delta_gain": "delta_chosen_domain",
            "n_domain_items": "n_chosen_domain_items",
        }
    )
    rj = rj.rename(
        columns={
            "chosen_gain": "chosen_gain_rejected_domain",
            "rejected_gain": "rejected_gain_rejected_domain",
            "v_delta_gain": "delta_rejected_domain",
            "n_domain_items": "n_rejected_domain_items",
        }
    )

    keep_ch = [
        "row_id",
        "prompt_hash",
        "chosen_gain_chosen_domain",
        "rejected_gain_chosen_domain",
        "delta_chosen_domain",
        "n_chosen_domain_items",
    ]
    keep_rj = [
        "row_id",
        "prompt_hash",
        "chosen_gain_rejected_domain",
        "rejected_gain_rejected_domain",
        "delta_rejected_domain",
        "n_rejected_domain_items",
    ]

    df = ch[keep_ch].merge(
        rj[keep_rj],
        on=["row_id", "prompt_hash"],
        how="inner",
        validate="one_to_one",
    )

    df["chosen_domain_specificity"] = (
        df["chosen_gain_chosen_domain"] - df["chosen_gain_rejected_domain"]
    )
    df["rejected_domain_specificity"] = (
        df["rejected_gain_chosen_domain"] - df["rejected_gain_rejected_domain"]
    )
    df["domain_contrast_delta"] = (
        df["chosen_domain_specificity"] - df["rejected_domain_specificity"]
    )

    # Equivalent to delta_chosen_domain - delta_rejected_domain.
    df["domain_contrast_delta_alt"] = df["delta_chosen_domain"] - df["delta_rejected_domain"]

    return df


def load_surface(surface_tag: str) -> pd.DataFrame:
    surface_path = OUT_DIR / f"litbench_surface_features_{surface_tag}.csv"
    if not surface_path.exists():
        raise FileNotFoundError(surface_path)
    return pd.read_csv(surface_path)


def make_pairwise_examples(df: pd.DataFrame, feature_cols: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []

    for _, row in df.iterrows():
        chosen = []
        rejected = []

        for base in feature_cols:
            if base == "domain_specificity":
                chosen.append(row["chosen_domain_specificity"])
                rejected.append(row["rejected_domain_specificity"])
            elif base == "chosen_domain_v":
                chosen.append(row["chosen_gain_chosen_domain"])
                rejected.append(row["rejected_gain_chosen_domain"])
            elif base == "rejected_domain_v":
                chosen.append(row["chosen_gain_rejected_domain"])
                rejected.append(row["rejected_gain_rejected_domain"])
            else:
                chosen.append(row[f"chosen_{base}"])
                rejected.append(row[f"rejected_{base}"])

        chosen = np.asarray(chosen, dtype=float)
        rejected = np.asarray(rejected, dtype=float)

        if rng.random() < 0.5:
            rows.append(chosen - rejected)
            labels.append(1)
        else:
            rows.append(rejected - chosen)
            labels.append(0)

    return np.vstack(rows), np.asarray(labels, dtype=int)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chosen-tag", default="test_prompt_v_distilgpt2_otherchosen_mindomain2_maxdomain3")
    ap.add_argument("--rejected-tag", default="test_prompt_v_distilgpt2_otherrejected_mindomain2_maxdomain3")
    ap.add_argument("--surface-tag", default="test_ids_complete")
    ap.add_argument("--tag", default="test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain3")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    contrast = load_scores(args.chosen_tag, args.rejected_tag)
    surface = load_surface(args.surface_tag)

    df = surface.merge(contrast, on="row_id", how="inner", validate="one_to_one")

    # Direct rules.
    direct_rules = {
        "domain_contrast_sign_rule": df["domain_contrast_delta"] > 0,
        "chosen_domain_sign_rule": df["delta_chosen_domain"] > 0,
        "rejected_domain_reversed_sign_rule": df["delta_rejected_domain"] < 0,
    }

    rows = []
    correctness = {}

    for name, pred in direct_rules.items():
        correct = pred.to_numpy(dtype=bool)
        correctness[name] = correct
        rows.append(
            {
                "tag": args.tag,
                "model": name,
                "features": json.dumps([name]),
                **bootstrap_accuracy(correct, n_boot=args.n_boot, seed=args.seed),
            }
        )

    feature_sets = {
        "surface_format": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
        ],
        "domain_specificity_logistic": [
            "domain_specificity",
        ],
        "surface_plus_domain_specificity": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
            "domain_specificity",
        ],
        "surface_plus_chosen_and_rejected_v": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
            "chosen_domain_v",
            "rejected_domain_v",
        ],
    }

    for name, features in feature_sets.items():
        X, y = make_pairwise_examples(df, features, seed=args.seed)
        valid = np.isfinite(X).all(axis=1)
        X = X[valid]
        y = y[valid]

        preds = crossval_predictions(X, y, n_splits=args.n_splits, seed=args.seed)
        correct = preds == y
        correctness[name] = correct

        rows.append(
            {
                "tag": args.tag,
                "model": name,
                "features": json.dumps(features),
                **bootstrap_accuracy(correct, n_boot=args.n_boot, seed=args.seed),
            }
        )

    comparisons = [
        ("domain_contrast_sign_rule", "surface_format"),
        ("domain_specificity_logistic", "surface_format"),
        ("surface_plus_domain_specificity", "surface_format"),
        ("surface_plus_domain_specificity", "domain_specificity_logistic"),
        ("surface_plus_chosen_and_rejected_v", "surface_format"),
    ]

    delta_rows = []
    for a, b in comparisons:
        delta_rows.append(
            {
                "tag": args.tag,
                "model_a": a,
                "model_b": b,
                **paired_bootstrap_delta(correctness[a], correctness[b], n_boot=args.n_boot, seed=args.seed),
            }
        )

    continuous_rows = []
    for col in [
        "domain_contrast_delta",
        "delta_chosen_domain",
        "delta_rejected_domain",
        "chosen_domain_specificity",
        "rejected_domain_specificity",
    ]:
        continuous_rows.append(
            {
                "tag": args.tag,
                "quantity": col,
                **bootstrap_mean(df[col].to_numpy(dtype=float), n_boot=args.n_boot, seed=args.seed),
            }
        )

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    deltas = pd.DataFrame(delta_rows)
    continuous = pd.DataFrame(continuous_rows)
    merged = df[
        [
            "row_id",
            "prompt_hash",
            "n_chosen_domain_items",
            "n_rejected_domain_items",
            "chosen_gain_chosen_domain",
            "rejected_gain_chosen_domain",
            "delta_chosen_domain",
            "chosen_gain_rejected_domain",
            "rejected_gain_rejected_domain",
            "delta_rejected_domain",
            "chosen_domain_specificity",
            "rejected_domain_specificity",
            "domain_contrast_delta",
        ]
    ].copy()

    summary_path = OUT_DIR / f"litbench_prompt_v_domain_contrast_models_{args.tag}.csv"
    deltas_path = OUT_DIR / f"litbench_prompt_v_domain_contrast_deltas_{args.tag}.csv"
    continuous_path = OUT_DIR / f"litbench_prompt_v_domain_contrast_continuous_{args.tag}.csv"
    scores_path = OUT_DIR / f"litbench_prompt_v_domain_contrast_scores_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_prompt_v_domain_contrast_manifest_{args.tag}.json"

    summary.to_csv(summary_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    continuous.to_csv(continuous_path, index=False)
    merged.to_csv(scores_path, index=False)

    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "Preferred-domain minus rejected-domain prompt-conditioned V contrast",
                "chosen_tag": args.chosen_tag,
                "rejected_tag": args.rejected_tag,
                "surface_tag": args.surface_tag,
                "tag": args.tag,
                "n_rows_overlap": int(len(df)),
                "n_splits": args.n_splits,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "outputs": {
                    "summary": str(summary_path),
                    "deltas": str(deltas_path),
                    "continuous": str(continuous_path),
                    "scores": str(scores_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print()
    print(deltas.to_string(index=False))
    print()
    print(continuous.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {deltas_path}")
    print(f"Wrote {continuous_path}")
    print(f"Wrote {scores_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
