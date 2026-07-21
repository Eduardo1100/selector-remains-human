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


def paired_bootstrap_delta(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
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


def load_inputs(surface_tag: str, v_tag: str) -> pd.DataFrame:
    surface_path = OUT_DIR / f"litbench_surface_features_{surface_tag}.csv"
    v_path = OUT_DIR / f"litbench_prompt_conditioned_v_scores_{v_tag}.csv"

    if not surface_path.exists():
        raise FileNotFoundError(surface_path)
    if not v_path.exists():
        raise FileNotFoundError(v_path)

    surface = pd.read_csv(surface_path)
    v = pd.read_csv(v_path)

    keep_v = [
        "row_id",
        "prompt_hash",
        "n_domain_items",
        "chosen_gain",
        "rejected_gain",
        "v_delta_gain",
        "v_delta_cond_nll",
        "chosen_wins_v",
        "mean_target_tokens",
        "mean_prompt_context_tokens",
        "mean_candidate_context_tokens",
    ]

    df = surface.merge(v[keep_v], on="row_id", how="inner", validate="one_to_one")
    return df


def make_pairwise_examples(df: pd.DataFrame, feature_cols: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []

    for _, row in df.iterrows():
        chosen = []
        rejected = []

        for base in feature_cols:
            if base == "prompt_conditioned_v":
                chosen.append(row["chosen_gain"])
                rejected.append(row["rejected_gain"])
            elif base == "prompt_conditioned_cond_nll":
                # lower conditioned NLL is better, so negate to make higher-is-better.
                chosen.append(-row["chosen_cond_avg_nll"] if "chosen_cond_avg_nll" in row else row["chosen_gain"])
                rejected.append(-row["rejected_cond_avg_nll"] if "rejected_cond_avg_nll" in row else row["rejected_gain"])
            else:
                chosen.append(row[f"chosen_{base}"])
                rejected.append(row[f"rejected_{base}"])

        chosen = np.asarray(chosen, dtype=float)
        rejected = np.asarray(rejected, dtype=float)

        # Randomize orientation to prevent position leakage.
        if rng.random() < 0.5:
            rows.append(chosen - rejected)
            labels.append(1)
        else:
            rows.append(rejected - chosen)
            labels.append(0)

    return np.vstack(rows), np.asarray(labels, dtype=int)


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


def sign_rule_correct(df: pd.DataFrame, feature: str) -> np.ndarray:
    if feature == "prompt_conditioned_v":
        return (df["chosen_gain"] > df["rejected_gain"]).to_numpy(dtype=bool)
    raise ValueError(feature)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface-tag", default="test_ids_complete")
    ap.add_argument("--v-tag", default="test_prompt_v_distilgpt2_otherchosen_mindomain2_maxdomain3")
    ap.add_argument("--tag", default="test_prompt_v_distilgpt2_otherchosen_mindomain2_maxdomain3")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = load_inputs(args.surface_tag, args.v_tag)

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
        "v_only_logistic": [
            "prompt_conditioned_v",
        ],
        "surface_plus_v": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
            "prompt_conditioned_v",
        ],
        "surface_plus_v_and_domain_meta": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
            "prompt_conditioned_v",
        ],
    }

    rows = []
    correctness_by_model = {}

    # Include the direct sign rule for interpretability.
    direct_correct = sign_rule_correct(df, "prompt_conditioned_v")
    correctness_by_model["v_only_sign_rule"] = direct_correct
    rows.append(
        {
            "tag": args.tag,
            "model": "v_only_sign_rule",
            "features": json.dumps(["chosen_gain > rejected_gain"]),
            **bootstrap_accuracy(direct_correct, n_boot=args.n_boot, seed=args.seed),
        }
    )

    for name, features in feature_sets.items():
        X, y = make_pairwise_examples(df, features, seed=args.seed)
        valid = np.isfinite(X).all(axis=1)
        X = X[valid]
        y = y[valid]

        preds = crossval_predictions(X, y, n_splits=args.n_splits, seed=args.seed)
        correct = preds == y

        correctness_by_model[name] = correct

        rows.append(
            {
                "tag": args.tag,
                "model": name,
                "features": json.dumps(features),
                **bootstrap_accuracy(correct, n_boot=args.n_boot, seed=args.seed),
            }
        )

    comparisons = [
        ("surface_plus_v", "surface_format"),
        ("surface_plus_v", "v_only_logistic"),
        ("surface_format", "v_only_logistic"),
        ("v_only_sign_rule", "surface_format"),
        ("surface_plus_v", "v_only_sign_rule"),
    ]

    delta_rows = []
    for a, b in comparisons:
        delta_rows.append(
            {
                "tag": args.tag,
                "model_a": a,
                "model_b": b,
                **paired_bootstrap_delta(
                    correctness_by_model[a],
                    correctness_by_model[b],
                    n_boot=args.n_boot,
                    seed=args.seed,
                ),
            }
        )

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    deltas = pd.DataFrame(delta_rows)

    summary_path = OUT_DIR / f"litbench_prompt_v_surface_control_models_{args.tag}.csv"
    deltas_path = OUT_DIR / f"litbench_prompt_v_surface_control_deltas_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_prompt_v_surface_control_manifest_{args.tag}.json"

    summary.to_csv(summary_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "Prompt-conditioned V controlled against formatting features",
                "surface_tag": args.surface_tag,
                "v_tag": args.v_tag,
                "tag": args.tag,
                "n_rows": int(len(df)),
                "n_splits": args.n_splits,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "outputs": {
                    "summary": str(summary_path),
                    "deltas": str(deltas_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print()
    print(deltas.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {deltas_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
