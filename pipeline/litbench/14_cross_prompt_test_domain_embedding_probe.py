from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"


SURFACE_FEATURES = [
    "chars",
    "words",
    "type_token_ratio",
    "avg_word_len",
    "punct_count",
    "newline_count",
    "paragraph_count",
]


def md5_text(x: object) -> str:
    return hashlib.md5(str(x).encode("utf-8")).hexdigest()


def pick_col(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Could not find {label}. Tried {list(candidates)}. Available: {list(df.columns)}")


def load_hf_dataset(name: str) -> pd.DataFrame:
    from datasets import load_dataset

    ds = load_dataset(name)
    split = next(iter(ds.keys()))
    return ds[split].to_pandas().reset_index(drop=True)


def encode_texts(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


def topk_mean(values: np.ndarray, k: int) -> float:
    if len(values) == 0:
        return float("nan")
    kk = min(k, len(values))
    idx = np.argpartition(values, -kk)[-kk:]
    return float(np.mean(values[idx]))


def domain_score(candidate_vec: np.ndarray, domain_vecs: np.ndarray, operator: str) -> float:
    sims = domain_vecs @ candidate_vec

    if operator == "mean":
        return float(np.mean(sims))
    if operator == "max":
        return float(np.max(sims))
    if operator == "top2":
        return topk_mean(sims, 2)
    if operator == "top3":
        return topk_mean(sims, 3)
    if operator == "top5":
        return topk_mean(sims, 5)

    raise ValueError(f"Unknown operator: {operator}")


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
        raise ValueError(f"Paired arrays differ: {len(a)} vs {len(b)}")

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


def pairwise_cv_accuracy(feature_diffs: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(feature_diffs, dtype=float)
    if X.ndim == 1:
        X = X[:, None]

    n = X.shape[0]
    correct = np.zeros(n, dtype=bool)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, test_idx in kf.split(X):
        X_train = np.vstack([X[train_idx], -X[train_idx]])
        y_train = np.concatenate(
            [np.ones(len(train_idx), dtype=int), np.zeros(len(train_idx), dtype=int)]
        )

        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, solver="lbfgs"),
        )
        clf.fit(X_train, y_train)
        pred = clf.predict(X[test_idx])
        correct[test_idx] = pred == 1

    return correct


def load_surface(surface_tag: str) -> pd.DataFrame:
    path = OUT_DIR / f"litbench_surface_features_{surface_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def surface_feature_matrix(surface: pd.DataFrame, row_ids: list[int]) -> np.ndarray:
    sub = surface[surface["row_id"].isin(row_ids)].copy()
    sub = sub.set_index("row_id").loc[row_ids].reset_index()

    cols = []
    for feat in SURFACE_FEATURES:
        chosen = f"chosen_{feat}"
        rejected = f"rejected_{feat}"
        if chosen not in sub.columns or rejected not in sub.columns:
            raise KeyError(f"Missing surface columns {chosen}/{rejected}")
        cols.append(sub[chosen].to_numpy(dtype=float) - sub[rejected].to_numpy(dtype=float))

    return np.vstack(cols).T


def load_row_filter(row_filter_domain_tag: str | None) -> set[int] | None:
    if not row_filter_domain_tag:
        return None

    path = OUT_DIR / f"litbench_prompt_v_domain_contrast_scores_{row_filter_domain_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    ids = set(pd.read_csv(path)["row_id"].astype(int).tolist())
    print(f"Loaded row filter: {len(ids)} row_ids from {path.name}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--surface-tag", default="test_ids_complete")
    ap.add_argument("--row-filter-domain-tag", default=None)
    ap.add_argument("--tag", default="cross_prompt_test_domain_embedding_min2_max10_top50")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embedding-batch-size", type=int, default=32)
    ap.add_argument("--limit-test", type=int, default=None)
    ap.add_argument("--top-source-prompts", type=int, default=50)
    ap.add_argument("--min-domain", type=int, default=2)
    ap.add_argument("--max-domain", type=int, default=10)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    print(f"Loading test dataset: {args.dataset}")
    df = load_hf_dataset(args.dataset)
    df = df.copy()
    df["__row_id"] = np.arange(len(df), dtype=int)

    row_filter_ids = load_row_filter(args.row_filter_domain_tag)
    if row_filter_ids is not None:
        before = len(df)
        df = df[df["__row_id"].isin(row_filter_ids)].copy().reset_index(drop=True)
        print(f"Restricting scored rows to {len(df)} / {before}")

    if args.limit_test is not None and len(df) > args.limit_test:
        before = len(df)
        df = df.iloc[: args.limit_test].copy().reset_index(drop=True)
        print(f"Limiting scored rows to {len(df)} / {before}")

    # Load full unfiltered table as source-domain pool.
    full_df = load_hf_dataset(args.dataset)
    full_df = full_df.copy()
    full_df["__row_id"] = np.arange(len(full_df), dtype=int)

    prompt_col = pick_col(full_df, ["prompt", "instruction", "question"], "prompt")
    chosen_col = pick_col(full_df, ["chosen_story", "chosen", "winner", "response_chosen"], "chosen story")
    rejected_col = pick_col(full_df, ["rejected_story", "rejected", "loser", "response_rejected"], "rejected story")

    full_prompts = full_df[prompt_col].astype(str).tolist()
    full_prompt_hashes = np.array([md5_text(p) for p in full_prompts])

    scored_prompts = df[prompt_col].astype(str).tolist()
    scored_prompt_hashes = np.array([md5_text(p) for p in scored_prompts])
    scored_row_ids = df["__row_id"].astype(int).to_numpy()

    full_chosen = full_df[chosen_col].astype(str).tolist()
    full_rejected = full_df[rejected_col].astype(str).tolist()

    scored_chosen = df[chosen_col].astype(str).tolist()
    scored_rejected = df[rejected_col].astype(str).tolist()

    print(f"Source test pairs: {len(full_df)}")
    print(f"Scored test pairs: {len(df)}")

    print("Encoding full test prompts")
    full_prompt_emb = encode_texts(full_prompts, args.embedding_model, args.embedding_batch_size)

    print("Encoding scored test prompts")
    scored_prompt_emb = encode_texts(scored_prompts, args.embedding_model, args.embedding_batch_size)

    print("Encoding full test chosen stories")
    full_chosen_emb = encode_texts(full_chosen, args.embedding_model, args.embedding_batch_size)

    print("Encoding full test rejected stories")
    full_rejected_emb = encode_texts(full_rejected, args.embedding_model, args.embedding_batch_size)

    print("Encoding scored chosen stories")
    scored_chosen_emb = encode_texts(scored_chosen, args.embedding_model, args.embedding_batch_size)

    print("Encoding scored rejected stories")
    scored_rejected_emb = encode_texts(scored_rejected, args.embedding_model, args.embedding_batch_size)

    operators = ["mean", "max", "top2", "top3", "top5"]
    rows = []

    top_pool = max(args.top_source_prompts, args.max_domain)

    for test_pos in tqdm(range(len(df)), desc="cross-prompt test-domain probe"):
        original_row_id = int(scored_row_ids[test_pos])
        prompt_hash = scored_prompt_hashes[test_pos]

        sims = full_prompt_emb @ scored_prompt_emb[test_pos]

        # Exclude the literal same prompt, not just the same row.
        valid = full_prompt_hashes != prompt_hash

        # Also exclude same row defensively.
        valid &= full_df["__row_id"].to_numpy(dtype=int) != original_row_id

        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) < args.min_domain:
            continue

        valid_sims = sims[valid_idx]
        kk = min(top_pool, len(valid_idx))
        near_local = np.argpartition(valid_sims, -kk)[-kk:]
        near_idx_unsorted = valid_idx[near_local]
        near_idx = near_idx_unsorted[np.argsort(sims[near_idx_unsorted])[::-1]]

        domain_idx = near_idx[: args.max_domain]

        if len(domain_idx) < args.min_domain:
            continue

        preferred_vecs = full_chosen_emb[domain_idx]
        rejected_vecs = full_rejected_emb[domain_idx]

        chosen_vec = scored_chosen_emb[test_pos]
        rejected_vec = scored_rejected_emb[test_pos]

        base = {
            "row_id": original_row_id,
            "prompt_hash": prompt_hash,
            "n_preferred_domain": int(len(preferred_vecs)),
            "n_rejected_domain": int(len(rejected_vecs)),
            "mean_prompt_sim": float(np.mean(sims[domain_idx])),
            "max_prompt_sim": float(np.max(sims[domain_idx])),
            "source_row_ids": json.dumps([int(x) for x in domain_idx.tolist()]),
        }

        for op in operators:
            chosen_to_pref = domain_score(chosen_vec, preferred_vecs, op)
            chosen_to_rej = domain_score(chosen_vec, rejected_vecs, op)
            rejected_to_pref = domain_score(rejected_vec, preferred_vecs, op)
            rejected_to_rej = domain_score(rejected_vec, rejected_vecs, op)

            chosen_spec = chosen_to_pref - chosen_to_rej
            rejected_spec = rejected_to_pref - rejected_to_rej
            delta = chosen_spec - rejected_spec

            rows.append(
                {
                    **base,
                    "operator": op,
                    "chosen_domain_specificity": chosen_spec,
                    "rejected_domain_specificity": rejected_spec,
                    "domain_contrast_delta": delta,
                    "correct_sign_rule": bool(delta > 0),
                    "chosen_to_preferred_test_other_prompt": chosen_to_pref,
                    "chosen_to_rejected_test_other_prompt": chosen_to_rej,
                    "rejected_to_preferred_test_other_prompt": rejected_to_pref,
                    "rejected_to_rejected_test_other_prompt": rejected_to_rej,
                }
            )

    scores = pd.DataFrame(rows)
    if scores.empty:
        raise SystemExit("No scores produced.")

    surface = load_surface(args.surface_tag)

    model_rows = []
    delta_rows = []
    continuous_rows = []

    for op, sub in scores.groupby("operator", sort=False):
        sub = sub.sort_values("row_id").reset_index(drop=True)
        row_ids = sub["row_id"].astype(int).tolist()

        y_sign = sub["correct_sign_rule"].to_numpy(dtype=bool)
        model_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model": f"{op}_cross_prompt_test_domain_sign_rule",
                "features": json.dumps(["domain_contrast_delta > 0"]),
                **bootstrap_accuracy(y_sign, args.n_boot, args.seed),
            }
        )

        X_delta = sub["domain_contrast_delta"].to_numpy(dtype=float)[:, None]
        y_log = pairwise_cv_accuracy(X_delta, args.n_splits, args.seed)
        model_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model": f"{op}_cross_prompt_test_domain_logistic",
                "features": json.dumps(["domain_contrast_delta"]),
                **bootstrap_accuracy(y_log, args.n_boot, args.seed),
            }
        )

        X_surface = surface_feature_matrix(surface, row_ids)
        y_surface = pairwise_cv_accuracy(X_surface, args.n_splits, args.seed)
        model_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model": "surface_format",
                "features": json.dumps(SURFACE_FEATURES),
                **bootstrap_accuracy(y_surface, args.n_boot, args.seed),
            }
        )

        X_surface_plus = np.column_stack([X_surface, X_delta])
        y_surface_plus = pairwise_cv_accuracy(X_surface_plus, args.n_splits, args.seed)
        model_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model": f"{op}_surface_plus_cross_prompt_test_domain",
                "features": json.dumps(SURFACE_FEATURES + ["domain_contrast_delta"]),
                **bootstrap_accuracy(y_surface_plus, args.n_boot, args.seed),
            }
        )

        for model_a, corr_a in [
            (f"{op}_cross_prompt_test_domain_sign_rule", y_sign),
            (f"{op}_cross_prompt_test_domain_logistic", y_log),
            (f"{op}_surface_plus_cross_prompt_test_domain", y_surface_plus),
        ]:
            delta_rows.append(
                {
                    "tag": args.tag,
                    "operator": op,
                    "model_a": model_a,
                    "model_b": "surface_format",
                    **paired_bootstrap_delta(corr_a, y_surface, args.n_boot, args.seed),
                }
            )

        delta_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model_a": f"{op}_surface_plus_cross_prompt_test_domain",
                "model_b": f"{op}_cross_prompt_test_domain_logistic",
                **paired_bootstrap_delta(y_surface_plus, y_log, args.n_boot, args.seed),
            }
        )

        for quantity in [
            "domain_contrast_delta",
            "chosen_domain_specificity",
            "rejected_domain_specificity",
            "mean_prompt_sim",
            "max_prompt_sim",
        ]:
            values = sub[quantity].to_numpy(dtype=float)
            rng = np.random.default_rng(args.seed)
            boots = np.empty(args.n_boot, dtype=float)

            for b in range(args.n_boot):
                idx = rng.integers(0, len(values), size=len(values))
                boots[b] = float(np.mean(values[idx]))

            continuous_rows.append(
                {
                    "tag": args.tag,
                    "operator": op,
                    "quantity": quantity,
                    "mean": float(np.mean(values)),
                    "ci95_low": float(np.quantile(boots, 0.025)),
                    "ci95_high": float(np.quantile(boots, 0.975)),
                    "p_le_zero": float(np.mean(boots <= 0.0)),
                    "p_ge_zero": float(np.mean(boots >= 0.0)),
                    "n": int(len(values)),
                    "n_boot": int(args.n_boot),
                }
            )

    models = pd.DataFrame(model_rows).sort_values("accuracy", ascending=False)
    deltas = pd.DataFrame(delta_rows)
    continuous = pd.DataFrame(continuous_rows)

    scores_path = OUT_DIR / f"litbench_cross_prompt_test_domain_embedding_scores_{args.tag}.csv"
    models_path = OUT_DIR / f"litbench_cross_prompt_test_domain_embedding_models_{args.tag}.csv"
    deltas_path = OUT_DIR / f"litbench_cross_prompt_test_domain_embedding_deltas_{args.tag}.csv"
    continuous_path = OUT_DIR / f"litbench_cross_prompt_test_domain_embedding_continuous_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_cross_prompt_test_domain_embedding_manifest_{args.tag}.json"

    scores.to_csv(scores_path, index=False)
    models.to_csv(models_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    continuous.to_csv(continuous_path, index=False)

    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "Different-prompt test-label domain probe",
                "dataset": args.dataset,
                "surface_tag": args.surface_tag,
                "row_filter_domain_tag": args.row_filter_domain_tag,
                "tag": args.tag,
                "embedding_model": args.embedding_model,
                "limit_test": args.limit_test,
                "top_source_prompts": args.top_source_prompts,
                "min_domain": args.min_domain,
                "max_domain": args.max_domain,
                "n_scores_rows": int(len(scores)),
                "n_unique_test_rows": int(scores["row_id"].nunique()),
                "operators": sorted(scores["operator"].unique().tolist()),
                "same_prompt_excluded": True,
                "n_splits": args.n_splits,
                "n_boot": args.n_boot,
                "seed": args.seed,
                "outputs": {
                    "scores": str(scores_path),
                    "models": str(models_path),
                    "deltas": str(deltas_path),
                    "continuous": str(continuous_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(models.to_string(index=False))
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
