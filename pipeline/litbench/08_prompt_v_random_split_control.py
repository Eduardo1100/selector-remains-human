from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PV_PATH = ROOT / "pipeline" / "litbench" / "05_prompt_conditioned_v.py"


def load_prompt_v_module():
    spec = importlib.util.spec_from_file_location("prompt_conditioned_v", PV_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {PV_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stable_seed(text: str, seed: int) -> int:
    h = hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()
    return int(h[:8], 16)


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


def assign_random_pools(story_table: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = story_table.copy()
    out["random_pool"] = ""
    out["random_rank"] = -1

    for prompt_hash, group in out.groupby("prompt_hash"):
        idxs = list(group.index)
        rng = np.random.default_rng(stable_seed(str(prompt_hash), seed))
        rng.shuffle(idxs)

        for rank, idx in enumerate(idxs):
            out.loc[idx, "random_pool"] = "A" if rank % 2 == 0 else "B"
            out.loc[idx, "random_rank"] = rank

    return out


def select_random_domain(
    story_table: pd.DataFrame,
    prompt_hash_value: str,
    current_row_id: int,
    pool: str,
    max_domain: int,
) -> pd.DataFrame:
    group = story_table[
        (story_table["prompt_hash"] == prompt_hash_value)
        & (story_table["row_id"] != current_row_id)
        & (story_table["random_pool"] == pool)
    ].copy()

    group = group.sort_values(["random_rank", "story_hash"]).head(max_domain)
    return group.reset_index(drop=True)


def crossval_predictions(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray | None:
    counts = np.bincount(y, minlength=2)
    max_splits = int(min(n_splits, len(y), counts.min()))

    if max_splits < 2:
        return None

    skf = StratifiedKFold(n_splits=max_splits, shuffle=True, random_state=seed)
    preds = np.zeros_like(y)

    for train_idx, test_idx in skf.split(X, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, C=1.0),
        )
        clf.fit(X[train_idx], y[train_idx])
        preds[test_idx] = clf.predict(X[test_idx])

    return preds


def load_surface(surface_tag: str) -> pd.DataFrame:
    path = OUT_DIR / f"litbench_surface_features_{surface_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def make_pairwise_examples(df: pd.DataFrame, feature_cols: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows = []
    labels = []

    for _, row in df.iterrows():
        chosen = []
        rejected = []

        for base in feature_cols:
            if base == "random_domain_specificity":
                chosen.append(row["chosen_domain_specificity"])
                rejected.append(row["rejected_domain_specificity"])
            elif base == "random_pool_a_v":
                chosen.append(row["chosen_gain_a"])
                rejected.append(row["rejected_gain_a"])
            elif base == "random_pool_b_v":
                chosen.append(row["chosen_gain_b"])
                rejected.append(row["rejected_gain_b"])
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


def run_models(df: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    direct_rules = {
        "random_domain_contrast_sign_rule": df["domain_contrast_delta"] > 0,
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
        "random_domain_specificity_logistic": [
            "random_domain_specificity",
        ],
        "surface_plus_random_domain_specificity": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
            "random_domain_specificity",
        ],
        "surface_plus_random_pool_a_b_v": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
            "random_pool_a_v",
            "random_pool_b_v",
        ],
    }

    for name, features in feature_sets.items():
        X, y = make_pairwise_examples(df, features, seed=args.seed)
        valid = np.isfinite(X).all(axis=1)
        X = X[valid]
        y = y[valid]

        preds = crossval_predictions(X, y, n_splits=args.n_splits, seed=args.seed)
        if preds is None:
            rows.append(
                {
                    "tag": args.tag,
                    "model": name,
                    "features": json.dumps(features),
                    "accuracy": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "n": int(len(y)),
                    "n_boot": int(args.n_boot),
                    "note": "skipped_cv_too_few_samples_or_class_members",
                }
            )
            continue

        correct = preds == y
        correctness[name] = correct

        rows.append(
            {
                "tag": args.tag,
                "model": name,
                "features": json.dumps(features),
                **bootstrap_accuracy(correct, n_boot=args.n_boot, seed=args.seed),
                "note": "",
            }
        )

    comparisons = [
        ("random_domain_contrast_sign_rule", "surface_format"),
        ("random_domain_specificity_logistic", "surface_format"),
        ("surface_plus_random_domain_specificity", "surface_format"),
        ("surface_plus_random_domain_specificity", "random_domain_specificity_logistic"),
    ]

    delta_rows = []
    for a, b in comparisons:
        if a not in correctness or b not in correctness:
            delta_rows.append(
                {
                    "tag": args.tag,
                    "model_a": a,
                    "model_b": b,
                    "delta_accuracy": float("nan"),
                    "delta_ci95_low": float("nan"),
                    "delta_ci95_high": float("nan"),
                    "p_delta_le_zero": float("nan"),
                    "p_delta_ge_zero": float("nan"),
                    "note": "skipped_delta_missing_model",
                }
            )
            continue

        delta_rows.append(
            {
                "tag": args.tag,
                "model_a": a,
                "model_b": b,
                **paired_bootstrap_delta(
                    correctness[a],
                    correctness[b],
                    n_boot=args.n_boot,
                    seed=args.seed,
                ),
                "note": "",
            }
        )

    continuous_rows = []
    for col in [
        "domain_contrast_delta",
        "delta_a",
        "delta_b",
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

    return (
        pd.DataFrame(rows).sort_values("accuracy", ascending=False),
        pd.DataFrame(delta_rows),
        pd.DataFrame(continuous_rows),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--surface-tag", default="test_ids_complete")
    ap.add_argument("--tag", default="test_prompt_v_distilgpt2_randomsplit_seed123_mindomain2_maxdomain3")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--min-domain", type=int, default=2)
    ap.add_argument("--max-domain", type=int, default=3)
    ap.add_argument("--max-context-tokens", type=int, default=512)
    ap.add_argument("--max-target-tokens", type=int, default=384)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    pv = load_prompt_v_module()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install with:\n"
            "  python -m pip install transformers torch tqdm\n"
        ) from exc

    df = pv.load_hf_dataset(args.dataset)
    df["prompt_norm"] = df["prompt"].map(pv.norm_prompt)
    df["prompt_hash"] = df["prompt"].map(pv.prompt_hash)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    story_table = pv.make_story_table(df)
    story_table = assign_random_pools(story_table, seed=args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    rows = []
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="random split V"):
        domain_a = select_random_domain(
            story_table=story_table,
            prompt_hash_value=row["prompt_hash"],
            current_row_id=int(i),
            pool="A",
            max_domain=args.max_domain,
        )
        domain_b = select_random_domain(
            story_table=story_table,
            prompt_hash_value=row["prompt_hash"],
            current_row_id=int(i),
            pool="B",
            max_domain=args.max_domain,
        )

        if len(domain_a) < args.min_domain or len(domain_b) < args.min_domain:
            skipped += 1
            continue

        prompt_only_a = pv.score_domain(
            prompt=row["prompt"],
            candidate_story=None,
            domain=domain_a,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        chosen_a = pv.score_domain(
            prompt=row["prompt"],
            candidate_story=row["chosen_story"],
            domain=domain_a,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        rejected_a = pv.score_domain(
            prompt=row["prompt"],
            candidate_story=row["rejected_story"],
            domain=domain_a,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )

        prompt_only_b = pv.score_domain(
            prompt=row["prompt"],
            candidate_story=None,
            domain=domain_b,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        chosen_b = pv.score_domain(
            prompt=row["prompt"],
            candidate_story=row["chosen_story"],
            domain=domain_b,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        rejected_b = pv.score_domain(
            prompt=row["prompt"],
            candidate_story=row["rejected_story"],
            domain=domain_b,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )

        chosen_gain_a = prompt_only_a["avg_nll"] - chosen_a["avg_nll"]
        rejected_gain_a = prompt_only_a["avg_nll"] - rejected_a["avg_nll"]
        chosen_gain_b = prompt_only_b["avg_nll"] - chosen_b["avg_nll"]
        rejected_gain_b = prompt_only_b["avg_nll"] - rejected_b["avg_nll"]

        chosen_domain_specificity = chosen_gain_a - chosen_gain_b
        rejected_domain_specificity = rejected_gain_a - rejected_gain_b
        domain_contrast_delta = chosen_domain_specificity - rejected_domain_specificity

        rows.append(
            {
                "row_id": int(i),
                "prompt_hash": row["prompt_hash"],
                "n_domain_a_items": int(len(domain_a)),
                "n_domain_b_items": int(len(domain_b)),
                "chosen_gain_a": chosen_gain_a,
                "rejected_gain_a": rejected_gain_a,
                "delta_a": chosen_gain_a - rejected_gain_a,
                "chosen_gain_b": chosen_gain_b,
                "rejected_gain_b": rejected_gain_b,
                "delta_b": chosen_gain_b - rejected_gain_b,
                "chosen_domain_specificity": chosen_domain_specificity,
                "rejected_domain_specificity": rejected_domain_specificity,
                "domain_contrast_delta": domain_contrast_delta,
                "chosen_wins_random_contrast": bool(domain_contrast_delta > 0),
            }
        )

    scores = pd.DataFrame(rows)
    if len(scores) == 0:
        raise SystemExit("No eligible rows. Lower --min-domain or inspect prompt grouping.")

    surface = load_surface(args.surface_tag)
    model_df = surface.merge(scores, on="row_id", how="inner", validate="one_to_one")

    summary, deltas, continuous = run_models(model_df, args)

    scores_path = OUT_DIR / f"litbench_prompt_v_random_split_scores_{args.tag}.csv"
    models_path = OUT_DIR / f"litbench_prompt_v_random_split_models_{args.tag}.csv"
    deltas_path = OUT_DIR / f"litbench_prompt_v_random_split_deltas_{args.tag}.csv"
    continuous_path = OUT_DIR / f"litbench_prompt_v_random_split_continuous_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_prompt_v_random_split_manifest_{args.tag}.json"

    scores.to_csv(scores_path, index=False)
    summary.to_csv(models_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    continuous.to_csv(continuous_path, index=False)

    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "same-form nonhuman random pool split control for prompt-conditioned V",
                "dataset": args.dataset,
                "surface_tag": args.surface_tag,
                "tag": args.tag,
                "model": args.model,
                "min_domain": args.min_domain,
                "max_domain": args.max_domain,
                "max_context_tokens": args.max_context_tokens,
                "max_target_tokens": args.max_target_tokens,
                "n_total_rows": int(len(df)),
                "n_eligible_rows": int(len(scores)),
                "n_skipped_rows": int(skipped),
                "seed": args.seed,
                "n_boot": args.n_boot,
                "outputs": {
                    "scores": str(scores_path),
                    "models": str(models_path),
                    "deltas": str(deltas_path),
                    "continuous": str(continuous_path),
                },
                "interpretation": (
                    "Random same-prompt pools A/B replace preference-defined chosen/rejected pools. "
                    "If random domain contrast predicts preference strongly, prior preferred/rejected contrast may reflect generic within-class geometry. "
                    "If random split collapses toward chance, preference-labeled domain construction carries signal."
                ),
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
    print(f"Wrote {scores_path}")
    print(f"Wrote {models_path}")
    print(f"Wrote {deltas_path}")
    print(f"Wrote {continuous_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
