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
    raise KeyError(f"Could not find {label}. Tried: {list(candidates)}. Available: {list(df.columns)}")


def load_hf_dataset(name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets with: python -m pip install datasets pyarrow") from exc

    ds = load_dataset(name)
    split = next(iter(ds.keys()))
    return ds[split].to_pandas().reset_index(drop=True)


def make_story_table(df: pd.DataFrame) -> pd.DataFrame:
    prompt_col = pick_col(df, ["prompt", "instruction", "question"], "prompt")
    chosen_col = pick_col(df, ["chosen_story", "chosen", "winner", "response_chosen"], "chosen story")
    rejected_col = pick_col(df, ["rejected_story", "rejected", "loser", "response_rejected"], "rejected story")

    chosen_upvote_col = None
    rejected_upvote_col = None
    for c in ["chosen_upvotes", "chosen_score", "winner_upvotes", "upvotes_chosen"]:
        if c in df.columns:
            chosen_upvote_col = c
            break
    for c in ["rejected_upvotes", "rejected_score", "loser_upvotes", "upvotes_rejected"]:
        if c in df.columns:
            rejected_upvote_col = c
            break

    rows = []
    for row_id, row in df.iterrows():
        prompt = str(row[prompt_col])
        prompt_hash = md5_text(prompt)

        chosen_story = str(row[chosen_col])
        rejected_story = str(row[rejected_col])

        rows.append(
            {
                "row_id": int(row_id),
                "side": "chosen",
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "story": chosen_story,
                "story_hash": md5_text(chosen_story),
                "upvotes": float(row[chosen_upvote_col]) if chosen_upvote_col else 0.0,
            }
        )
        rows.append(
            {
                "row_id": int(row_id),
                "side": "rejected",
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "story": rejected_story,
                "story_hash": md5_text(rejected_story),
                "upvotes": float(row[rejected_upvote_col]) if rejected_upvote_col else 0.0,
            }
        )

    story_table = pd.DataFrame(rows).reset_index(drop=False).rename(columns={"index": "story_idx"})
    return story_table


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


def pairwise_cv_accuracy(
    feature_diffs: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

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


def knn_specificity(
    candidate_vec: np.ndarray,
    preferred_vecs: np.ndarray,
    rejected_vecs: np.ndarray,
    k: int,
) -> float:
    pref_sims = preferred_vecs @ candidate_vec
    rej_sims = rejected_vecs @ candidate_vec

    sims = np.concatenate([pref_sims, rej_sims])
    labels = np.concatenate(
        [np.ones(len(pref_sims), dtype=float), -np.ones(len(rej_sims), dtype=float)]
    )

    kk = min(k, len(sims))
    idx = np.argpartition(sims, -kk)[-kk:]
    return float(np.mean(labels[idx]))


def encode_stories(stories: list[str], model_name: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    emb = model.encode(
        stories,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(emb, dtype=np.float32)


def build_scores(
    df: pd.DataFrame,
    story_table: pd.DataFrame,
    embeddings: np.ndarray,
    min_domain: int,
    max_domain: int,
    limit: int | None,
    row_filter_ids: set[int] | None,
) -> pd.DataFrame:
    prompt_col = pick_col(df, ["prompt", "instruction", "question"], "prompt")
    rows = []

    grouped = {
        key: group.copy()
        for key, group in story_table.groupby(["prompt_hash", "side"], sort=False)
    }

    n_rows = len(df) if limit is None else min(limit, len(df))

    for row_id in tqdm(range(n_rows), desc="embedding operator scores"):
        if row_filter_ids is not None and row_id not in row_filter_ids:
            continue
        prompt_hash = md5_text(str(df.iloc[row_id][prompt_col]))

        chosen_pool = grouped.get((prompt_hash, "chosen"), pd.DataFrame())
        rejected_pool = grouped.get((prompt_hash, "rejected"), pd.DataFrame())

        chosen_domain = chosen_pool[chosen_pool["row_id"] != row_id].copy()
        rejected_domain = rejected_pool[rejected_pool["row_id"] != row_id].copy()

        chosen_domain = chosen_domain.sort_values(["upvotes", "row_id"], ascending=[False, True]).head(max_domain)
        rejected_domain = rejected_domain.sort_values(["upvotes", "row_id"], ascending=[False, True]).head(max_domain)

        if len(chosen_domain) < min_domain or len(rejected_domain) < min_domain:
            continue

        chosen_story_row = story_table[(story_table["row_id"] == row_id) & (story_table["side"] == "chosen")]
        rejected_story_row = story_table[(story_table["row_id"] == row_id) & (story_table["side"] == "rejected")]

        if len(chosen_story_row) != 1 or len(rejected_story_row) != 1:
            raise RuntimeError(f"Expected one chosen/rejected story row for row_id={row_id}")

        chosen_vec = embeddings[int(chosen_story_row.iloc[0]["story_idx"])]
        rejected_vec = embeddings[int(rejected_story_row.iloc[0]["story_idx"])]

        chosen_domain_vecs = embeddings[chosen_domain["story_idx"].to_numpy(dtype=int)]
        rejected_domain_vecs = embeddings[rejected_domain["story_idx"].to_numpy(dtype=int)]

        base = {
            "row_id": int(row_id),
            "prompt_hash": prompt_hash,
            "n_chosen_domain": int(len(chosen_domain)),
            "n_rejected_domain": int(len(rejected_domain)),
        }

        for op in ["mean", "max", "top2", "top3", "top5"]:
            chosen_to_chosen = domain_score(chosen_vec, chosen_domain_vecs, op)
            chosen_to_rejected = domain_score(chosen_vec, rejected_domain_vecs, op)
            rejected_to_chosen = domain_score(rejected_vec, chosen_domain_vecs, op)
            rejected_to_rejected = domain_score(rejected_vec, rejected_domain_vecs, op)

            chosen_spec = chosen_to_chosen - chosen_to_rejected
            rejected_spec = rejected_to_chosen - rejected_to_rejected
            delta = chosen_spec - rejected_spec

            rows.append(
                {
                    **base,
                    "operator": op,
                    "chosen_domain_specificity": chosen_spec,
                    "rejected_domain_specificity": rejected_spec,
                    "domain_contrast_delta": delta,
                    "correct_sign_rule": bool(delta > 0),
                    "chosen_to_chosen": chosen_to_chosen,
                    "chosen_to_rejected": chosen_to_rejected,
                    "rejected_to_chosen": rejected_to_chosen,
                    "rejected_to_rejected": rejected_to_rejected,
                }
            )

        for k in [3, 5]:
            op = f"knn{k}"
            chosen_spec = knn_specificity(chosen_vec, chosen_domain_vecs, rejected_domain_vecs, k)
            rejected_spec = knn_specificity(rejected_vec, chosen_domain_vecs, rejected_domain_vecs, k)
            delta = chosen_spec - rejected_spec

            rows.append(
                {
                    **base,
                    "operator": op,
                    "chosen_domain_specificity": chosen_spec,
                    "rejected_domain_specificity": rejected_spec,
                    "domain_contrast_delta": delta,
                    "correct_sign_rule": bool(delta > 0),
                    "chosen_to_chosen": np.nan,
                    "chosen_to_rejected": np.nan,
                    "rejected_to_chosen": np.nan,
                    "rejected_to_rejected": np.nan,
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--surface-tag", default="test_ids_complete")
    ap.add_argument("--row-filter-domain-tag", default=None, help="Optional domain-contrast tag whose score rows define the row_id subset to score.")
    ap.add_argument("--tag", default="test_prompt_domain_embedding_operators_mindomain2_maxdomain10")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embedding-batch-size", type=int, default=32)
    ap.add_argument("--min-domain", type=int, default=2)
    ap.add_argument("--max-domain", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    print("Loading dataset")
    df = load_hf_dataset(args.dataset)

    row_filter_ids = None
    if args.row_filter_domain_tag:
        filter_path = OUT_DIR / f"litbench_prompt_v_domain_contrast_scores_{args.row_filter_domain_tag}.csv"
        if not filter_path.exists():
            raise FileNotFoundError(filter_path)
        row_filter_ids = set(pd.read_csv(filter_path)["row_id"].astype(int).tolist())
        print(f"Restricting to {len(row_filter_ids)} row_ids from {filter_path.name}")

    print("Building story table")
    story_table = make_story_table(df)

    print(f"Encoding {len(story_table)} story instances with {args.embedding_model}")
    embeddings = encode_stories(
        story_table["story"].astype(str).tolist(),
        model_name=args.embedding_model,
        batch_size=args.embedding_batch_size,
    )

    print("Scoring per-story embedding operators")
    scores = build_scores(
        df=df,
        story_table=story_table,
        embeddings=embeddings,
        min_domain=args.min_domain,
        max_domain=args.max_domain,
        limit=args.limit,
        row_filter_ids=row_filter_ids,
    )

    if scores.empty:
        raise SystemExit("No eligible rows found.")

    surface = load_surface(args.surface_tag)

    model_rows = []
    delta_rows = []
    continuous_rows = []
    correctness_by_name = {}

    for op, sub in scores.groupby("operator", sort=False):
        sub = sub.sort_values("row_id").reset_index(drop=True)
        row_ids = sub["row_id"].astype(int).tolist()

        y_sign = sub["correct_sign_rule"].to_numpy(dtype=bool)
        correctness_by_name[f"{op}_domain_contrast_sign_rule"] = y_sign

        model_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model": f"{op}_domain_contrast_sign_rule",
                "features": json.dumps(["domain_contrast_delta > 0"]),
                **bootstrap_accuracy(y_sign, args.n_boot, args.seed),
            }
        )

        X_delta = sub["domain_contrast_delta"].to_numpy(dtype=float)[:, None]
        y_log = pairwise_cv_accuracy(X_delta, n_splits=args.n_splits, seed=args.seed)
        correctness_by_name[f"{op}_domain_specificity_logistic"] = y_log

        model_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model": f"{op}_domain_specificity_logistic",
                "features": json.dumps(["domain_contrast_delta"]),
                **bootstrap_accuracy(y_log, args.n_boot, args.seed),
            }
        )

        X_surface = surface_feature_matrix(surface, row_ids)
        y_surface = pairwise_cv_accuracy(X_surface, n_splits=args.n_splits, seed=args.seed)
        surface_name = f"{op}_surface_format"
        correctness_by_name[surface_name] = y_surface

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
        y_surface_plus = pairwise_cv_accuracy(X_surface_plus, n_splits=args.n_splits, seed=args.seed)
        correctness_by_name[f"{op}_surface_plus_domain_specificity"] = y_surface_plus

        model_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model": f"{op}_surface_plus_domain_specificity",
                "features": json.dumps(SURFACE_FEATURES + ["domain_contrast_delta"]),
                **bootstrap_accuracy(y_surface_plus, args.n_boot, args.seed),
            }
        )

        for name_a, corr_a in [
            (f"{op}_domain_contrast_sign_rule", y_sign),
            (f"{op}_domain_specificity_logistic", y_log),
            (f"{op}_surface_plus_domain_specificity", y_surface_plus),
        ]:
            delta_rows.append(
                {
                    "tag": args.tag,
                    "operator": op,
                    "model_a": name_a,
                    "model_b": "surface_format",
                    **paired_bootstrap_delta(corr_a, y_surface, args.n_boot, args.seed),
                }
            )

        delta_rows.append(
            {
                "tag": args.tag,
                "operator": op,
                "model_a": f"{op}_surface_plus_domain_specificity",
                "model_b": f"{op}_domain_specificity_logistic",
                **paired_bootstrap_delta(y_surface_plus, y_log, args.n_boot, args.seed),
            }
        )

        for quantity in [
            "domain_contrast_delta",
            "chosen_domain_specificity",
            "rejected_domain_specificity",
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

    models = pd.DataFrame(model_rows).sort_values(["accuracy"], ascending=False)
    deltas = pd.DataFrame(delta_rows)
    continuous = pd.DataFrame(continuous_rows)

    scores_path = OUT_DIR / f"litbench_prompt_domain_embedding_operator_scores_{args.tag}.csv"
    models_path = OUT_DIR / f"litbench_prompt_domain_embedding_operator_models_{args.tag}.csv"
    deltas_path = OUT_DIR / f"litbench_prompt_domain_embedding_operator_deltas_{args.tag}.csv"
    continuous_path = OUT_DIR / f"litbench_prompt_domain_embedding_operator_continuous_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_prompt_domain_embedding_operator_manifest_{args.tag}.json"

    scores.to_csv(scores_path, index=False)
    models.to_csv(models_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    continuous.to_csv(continuous_path, index=False)

    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "Crossed embedding operator control for LitBench preference-labeled domains",
                "dataset": args.dataset,
                "surface_tag": args.surface_tag,
                "tag": args.tag,
                "embedding_model": args.embedding_model,
                "min_domain": args.min_domain,
                "max_domain": args.max_domain,
                "limit": args.limit,
                "row_filter_domain_tag": args.row_filter_domain_tag,
                "n_eligible_rows": int(scores["row_id"].nunique()),
                "operators": sorted(scores["operator"].unique().tolist()),
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
