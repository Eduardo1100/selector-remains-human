from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_RE = re.compile(r"\s+")


def norm_prompt(s: str) -> str:
    return PROMPT_RE.sub(" ", str(s).strip().lower())


def prompt_hash(s: str) -> str:
    return hashlib.sha1(norm_prompt(s).encode("utf-8")).hexdigest()[:16]


def story_hash(s: str) -> str:
    return hashlib.sha1(str(s).encode("utf-8")).hexdigest()[:16]


def load_hf_dataset(name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with:\n"
            "  python -m pip install datasets pyarrow\n"
        ) from exc

    ds = load_dataset(name)
    split = next(iter(ds.keys()))
    return ds[split].to_pandas()


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


def make_story_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for i, row in df.iterrows():
        for side in ["chosen", "rejected"]:
            text = str(row[f"{side}_story"])
            rows.append(
                {
                    "row_id": int(i),
                    "side": side,
                    "prompt": row["prompt"],
                    "prompt_hash": row["prompt_hash"],
                    "story": text,
                    "story_hash": story_hash(text),
                    "upvotes": row.get(f"{side}_upvotes", np.nan),
                    "comment_id": row.get(f"{side}_comment_id", ""),
                    "reddit_post_id": row.get(f"{side}_reddit_post_id", ""),
                }
            )

    return pd.DataFrame(rows).drop_duplicates(["prompt_hash", "story_hash"]).reset_index(drop=True)


def select_domain(
    story_table: pd.DataFrame,
    prompt_hash_value: str,
    current_row_id: int,
    domain_mode: str,
    max_domain: int,
) -> pd.DataFrame:
    group = story_table[story_table["prompt_hash"] == prompt_hash_value].copy()
    group = group[group["row_id"] != current_row_id]

    if domain_mode == "other_chosen":
        group = group[group["side"] == "chosen"]
    elif domain_mode == "other_rejected":
        group = group[group["side"] == "rejected"]
    else:
        raise ValueError(domain_mode)

    return group.sort_values(["upvotes", "row_id"], ascending=[False, True]).head(max_domain).reset_index(drop=True)


def unit_mean_sparse(mat) -> sparse.csr_matrix:
    centroid = mat.mean(axis=0)
    if not sparse.issparse(centroid):
        centroid = sparse.csr_matrix(centroid)
    return normalize(centroid)


def unit_mean_dense(mat: np.ndarray) -> np.ndarray:
    v = np.asarray(mat, dtype=np.float32).mean(axis=0)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def sparse_cos_to_centroid(vec, centroid) -> float:
    value = vec @ centroid.T
    if hasattr(value, "toarray"):
        return float(value.toarray()[0, 0])
    return float(value)


def dense_cos_to_centroid(vec: np.ndarray, centroid: np.ndarray) -> float:
    return float(np.dot(vec, centroid))


def build_tfidf_vectors(story_table: pd.DataFrame, max_features: int, min_df: int):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=min_df,
        max_df=0.95,
        max_features=max_features,
        sublinear_tf=True,
        norm="l2",
    )

    texts = story_table["story"].astype(str).tolist()
    mat = vectorizer.fit_transform(texts)
    return vectorizer, mat


def build_embedding_vectors(story_table: pd.DataFrame, model_name: str, batch_size: int, device: str | None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: sentence-transformers. Install with:\n"
            "  python -m pip install sentence-transformers\n"
            "or rerun with --skip-embeddings"
        ) from exc

    model = SentenceTransformer(model_name, device=device)
    emb = model.encode(
        story_table["story"].astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(emb, dtype=np.float32)


def compute_kernel_scores(
    df: pd.DataFrame,
    story_table: pd.DataFrame,
    kernel_name: str,
    vectors,
    min_domain: int,
    max_domain: int,
) -> pd.DataFrame:
    rows = []

    for i, row in df.iterrows():
        chosen_domain = select_domain(
            story_table,
            prompt_hash_value=row["prompt_hash"],
            current_row_id=int(i),
            domain_mode="other_chosen",
            max_domain=max_domain,
        )
        rejected_domain = select_domain(
            story_table,
            prompt_hash_value=row["prompt_hash"],
            current_row_id=int(i),
            domain_mode="other_rejected",
            max_domain=max_domain,
        )

        if len(chosen_domain) < min_domain or len(rejected_domain) < min_domain:
            continue

        chosen_story_h = story_hash(row["chosen_story"])
        rejected_story_h = story_hash(row["rejected_story"])

        # Candidate rows should exist because story_table was built from the same df.
        chosen_idx = story_table.index[
            (story_table["prompt_hash"] == row["prompt_hash"])
            & (story_table["story_hash"] == chosen_story_h)
        ][0]
        rejected_idx = story_table.index[
            (story_table["prompt_hash"] == row["prompt_hash"])
            & (story_table["story_hash"] == rejected_story_h)
        ][0]

        chosen_domain_idxs = chosen_domain.index.to_numpy()
        rejected_domain_idxs = rejected_domain.index.to_numpy()

        if kernel_name == "tfidf":
            chosen_vec = vectors[chosen_idx]
            rejected_vec = vectors[rejected_idx]
            chosen_centroid = unit_mean_sparse(vectors[chosen_domain_idxs])
            rejected_centroid = unit_mean_sparse(vectors[rejected_domain_idxs])

            chosen_to_chosen = sparse_cos_to_centroid(chosen_vec, chosen_centroid)
            chosen_to_rejected = sparse_cos_to_centroid(chosen_vec, rejected_centroid)
            rejected_to_chosen = sparse_cos_to_centroid(rejected_vec, chosen_centroid)
            rejected_to_rejected = sparse_cos_to_centroid(rejected_vec, rejected_centroid)

        else:
            chosen_vec = vectors[chosen_idx]
            rejected_vec = vectors[rejected_idx]
            chosen_centroid = unit_mean_dense(vectors[chosen_domain_idxs])
            rejected_centroid = unit_mean_dense(vectors[rejected_domain_idxs])

            chosen_to_chosen = dense_cos_to_centroid(chosen_vec, chosen_centroid)
            chosen_to_rejected = dense_cos_to_centroid(chosen_vec, rejected_centroid)
            rejected_to_chosen = dense_cos_to_centroid(rejected_vec, chosen_centroid)
            rejected_to_rejected = dense_cos_to_centroid(rejected_vec, rejected_centroid)

        chosen_domain_specificity = chosen_to_chosen - chosen_to_rejected
        rejected_domain_specificity = rejected_to_chosen - rejected_to_rejected
        domain_contrast_delta = chosen_domain_specificity - rejected_domain_specificity

        rows.append(
            {
                "row_id": int(i),
                "prompt_hash": row["prompt_hash"],
                "kernel": kernel_name,
                "n_chosen_domain_items": int(len(chosen_domain)),
                "n_rejected_domain_items": int(len(rejected_domain)),
                "chosen_to_chosen_domain": chosen_to_chosen,
                "chosen_to_rejected_domain": chosen_to_rejected,
                "rejected_to_chosen_domain": rejected_to_chosen,
                "rejected_to_rejected_domain": rejected_to_rejected,
                "chosen_domain_specificity": chosen_domain_specificity,
                "rejected_domain_specificity": rejected_domain_specificity,
                "domain_contrast_delta": domain_contrast_delta,
                "chosen_wins_domain_contrast": bool(domain_contrast_delta > 0),
            }
        )

    return pd.DataFrame(rows)


def crossval_predictions(X: np.ndarray, y: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    counts = np.bincount(y, minlength=2)
    max_splits = int(min(n_splits, len(y), counts.min()))

    if max_splits < 2:
        raise ValueError("Too few samples or class members for cross-validation.")

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
            if base == "kernel_domain_specificity":
                chosen.append(row["chosen_domain_specificity"])
                rejected.append(row["rejected_domain_specificity"])
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


def evaluate_kernel(scores: pd.DataFrame, surface: pd.DataFrame, tag: str, n_splits: int, n_boot: int, seed: int):
    df = surface.merge(scores, on="row_id", how="inner", validate="one_to_one")

    rows = []
    correctness = {}

    direct_correct = (df["domain_contrast_delta"] > 0).to_numpy(dtype=bool)
    correctness["domain_contrast_sign_rule"] = direct_correct
    rows.append(
        {
            "tag": tag,
            "kernel": str(df["kernel"].iloc[0]),
            "model": "domain_contrast_sign_rule",
            "features": json.dumps(["domain_contrast_delta > 0"]),
            **bootstrap_accuracy(direct_correct, n_boot=n_boot, seed=seed),
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
            "kernel_domain_specificity",
        ],
        "surface_plus_domain_specificity": [
            "chars",
            "words",
            "type_token_ratio",
            "avg_word_len",
            "punct_count",
            "newline_count",
            "paragraph_count",
            "kernel_domain_specificity",
        ],
    }

    for name, features in feature_sets.items():
        X, y = make_pairwise_examples(df, features, seed=seed)
        valid = np.isfinite(X).all(axis=1)
        X = X[valid]
        y = y[valid]

        preds = crossval_predictions(X, y, n_splits=n_splits, seed=seed)
        correct = preds == y
        correctness[name] = correct

        rows.append(
            {
                "tag": tag,
                "kernel": str(df["kernel"].iloc[0]),
                "model": name,
                "features": json.dumps(features),
                **bootstrap_accuracy(correct, n_boot=n_boot, seed=seed),
            }
        )

    comparisons = [
        ("domain_contrast_sign_rule", "surface_format"),
        ("domain_specificity_logistic", "surface_format"),
        ("surface_plus_domain_specificity", "surface_format"),
        ("surface_plus_domain_specificity", "domain_specificity_logistic"),
    ]

    delta_rows = []
    for a, b in comparisons:
        delta_rows.append(
            {
                "tag": tag,
                "kernel": str(df["kernel"].iloc[0]),
                "model_a": a,
                "model_b": b,
                **paired_bootstrap_delta(
                    correctness[a],
                    correctness[b],
                    n_boot=n_boot,
                    seed=seed,
                ),
            }
        )

    continuous_rows = []
    for col in [
        "domain_contrast_delta",
        "chosen_domain_specificity",
        "rejected_domain_specificity",
    ]:
        continuous_rows.append(
            {
                "tag": tag,
                "kernel": str(df["kernel"].iloc[0]),
                "quantity": col,
                **bootstrap_mean(df[col].to_numpy(dtype=float), n_boot=n_boot, seed=seed),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(delta_rows), pd.DataFrame(continuous_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--surface-tag", default="test_ids_complete")
    ap.add_argument("--tag", default="test_prompt_domain_kernel_controls_mindomain2_maxdomain3")
    ap.add_argument("--min-domain", type=int, default=2)
    ap.add_argument("--max-domain", type=int, default=3)
    ap.add_argument("--tfidf-max-features", type=int, default=50000)
    ap.add_argument("--tfidf-min-df", type=int, default=2)
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embedding-batch-size", type=int, default=32)
    ap.add_argument("--embedding-device", default=None)
    ap.add_argument("--skip-embeddings", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = load_hf_dataset(args.dataset)
    df["prompt_norm"] = df["prompt"].map(norm_prompt)
    df["prompt_hash"] = df["prompt"].map(prompt_hash)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    story_table = make_story_table(df)

    print("Building TF-IDF vectors")
    _, tfidf_mat = build_tfidf_vectors(
        story_table,
        max_features=args.tfidf_max_features,
        min_df=args.tfidf_min_df,
    )

    print("Scoring TF-IDF preferred/rejected domain contrast")
    tfidf_scores = compute_kernel_scores(
        df=df,
        story_table=story_table,
        kernel_name="tfidf",
        vectors=tfidf_mat,
        min_domain=args.min_domain,
        max_domain=args.max_domain,
    )

    score_frames = [tfidf_scores]

    if not args.skip_embeddings:
        print(f"Building embedding vectors with {args.embedding_model}")
        emb = build_embedding_vectors(
            story_table,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
            device=args.embedding_device,
        )

        print("Scoring embedding preferred/rejected domain contrast")
        emb_scores = compute_kernel_scores(
            df=df,
            story_table=story_table,
            kernel_name="embedding",
            vectors=emb,
            min_domain=args.min_domain,
            max_domain=args.max_domain,
        )
        score_frames.append(emb_scores)

    all_scores = pd.concat(score_frames, ignore_index=True)
    surface = load_surface(args.surface_tag)

    model_frames = []
    delta_frames = []
    continuous_frames = []

    for kernel, group in all_scores.groupby("kernel"):
        models, deltas, continuous = evaluate_kernel(
            group,
            surface=surface,
            tag=args.tag,
            n_splits=args.n_splits,
            n_boot=args.n_boot,
            seed=args.seed,
        )
        model_frames.append(models)
        delta_frames.append(deltas)
        continuous_frames.append(continuous)

    models = pd.concat(model_frames, ignore_index=True).sort_values(["kernel", "accuracy"], ascending=[True, False])
    deltas = pd.concat(delta_frames, ignore_index=True)
    continuous = pd.concat(continuous_frames, ignore_index=True)

    scores_path = OUT_DIR / f"litbench_prompt_domain_kernel_scores_{args.tag}.csv"
    models_path = OUT_DIR / f"litbench_prompt_domain_kernel_models_{args.tag}.csv"
    deltas_path = OUT_DIR / f"litbench_prompt_domain_kernel_deltas_{args.tag}.csv"
    continuous_path = OUT_DIR / f"litbench_prompt_domain_kernel_continuous_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_prompt_domain_kernel_manifest_{args.tag}.json"

    all_scores.to_csv(scores_path, index=False)
    models.to_csv(models_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    continuous.to_csv(continuous_path, index=False)

    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "TF-IDF and embedding kernels on identical preferred/rejected same-prompt pools",
                "dataset": args.dataset,
                "surface_tag": args.surface_tag,
                "tag": args.tag,
                "min_domain": args.min_domain,
                "max_domain": args.max_domain,
                "tfidf_max_features": args.tfidf_max_features,
                "tfidf_min_df": args.tfidf_min_df,
                "embedding_model": None if args.skip_embeddings else args.embedding_model,
                "limit": args.limit,
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
