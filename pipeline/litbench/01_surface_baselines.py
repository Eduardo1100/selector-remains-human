from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    return WORD_RE.findall(str(text).lower())


def text_features(text: str) -> dict[str, float]:
    s = str(text)
    toks = tokenize(s)
    n_words = len(toks)
    unique_words = len(set(toks))

    return {
        "chars": float(len(s)),
        "words": float(n_words),
        "type_token_ratio": float(unique_words / n_words) if n_words else np.nan,
        "avg_word_len": float(np.mean([len(t) for t in toks])) if toks else np.nan,
        "punct_count": float(sum(1 for ch in s if not ch.isalnum() and not ch.isspace())),
        "newline_count": float(s.count("\n")),
        "paragraph_count": float(max(1, s.count("\n\n") + 1)),
    }


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


def load_hf_dataset(dataset_name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with:\n"
            "  python -m pip install datasets pyarrow\n"
        ) from exc

    ds = load_dataset(dataset_name)
    split = next(iter(ds.keys()))
    return ds[split].to_pandas()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--tag", default="test_ids_complete")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = load_hf_dataset(args.dataset)

    required = {"chosen_story", "rejected_story"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset lacks required columns: {sorted(missing)}")

    feature_rows = []
    for i, row in df.iterrows():
        chosen = text_features(row["chosen_story"])
        rejected = text_features(row["rejected_story"])

        out = {"row_id": i}
        for k, v in chosen.items():
            out[f"chosen_{k}"] = v
        for k, v in rejected.items():
            out[f"rejected_{k}"] = v

        for optional in [
            "chosen_upvotes",
            "rejected_upvotes",
            "chosen_comment_id",
            "rejected_comment_id",
            "chosen_reddit_post_id",
            "rejected_reddit_post_id",
        ]:
            if optional in df.columns:
                out[optional] = row[optional]

        feature_rows.append(out)

    feat = pd.DataFrame(feature_rows)

    tests = {
        "prefer_more_chars": feat["chosen_chars"] > feat["rejected_chars"],
        "prefer_fewer_chars": feat["chosen_chars"] < feat["rejected_chars"],
        "prefer_more_words": feat["chosen_words"] > feat["rejected_words"],
        "prefer_fewer_words": feat["chosen_words"] < feat["rejected_words"],
        "prefer_higher_ttr": feat["chosen_type_token_ratio"] > feat["rejected_type_token_ratio"],
        "prefer_lower_ttr": feat["chosen_type_token_ratio"] < feat["rejected_type_token_ratio"],
        "prefer_higher_avg_word_len": feat["chosen_avg_word_len"] > feat["rejected_avg_word_len"],
        "prefer_lower_avg_word_len": feat["chosen_avg_word_len"] < feat["rejected_avg_word_len"],
        "prefer_more_punct": feat["chosen_punct_count"] > feat["rejected_punct_count"],
        "prefer_less_punct": feat["chosen_punct_count"] < feat["rejected_punct_count"],
        "prefer_more_newlines": feat["chosen_newline_count"] > feat["rejected_newline_count"],
        "prefer_fewer_newlines": feat["chosen_newline_count"] < feat["rejected_newline_count"],
        "prefer_more_paragraphs": feat["chosen_paragraph_count"] > feat["rejected_paragraph_count"],
        "prefer_fewer_paragraphs": feat["chosen_paragraph_count"] < feat["rejected_paragraph_count"],
    }

    if {"chosen_upvotes", "rejected_upvotes"} <= set(feat.columns):
        tests["prefer_more_upvotes_sanity"] = feat["chosen_upvotes"] > feat["rejected_upvotes"]

    rows = []
    for name, pred in tests.items():
        correct = pred.to_numpy(dtype=bool)
        stat = bootstrap_accuracy(correct, n_boot=args.n_boot, seed=args.seed)
        rows.append(
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "baseline": name,
                **stat,
            }
        )

    summary = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    summary_path = OUT_DIR / f"litbench_surface_baselines_{args.tag}.csv"
    features_path = OUT_DIR / f"litbench_surface_features_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_surface_baselines_{args.tag}_manifest.json"

    summary.to_csv(summary_path, index=False)
    feat.to_csv(features_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "analysis": "surface-only pairwise baselines",
                "n_rows": int(len(df)),
                "outputs": {
                    "summary": str(summary_path),
                    "features": str(features_path),
                },
                "n_boot": args.n_boot,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Wrote {summary_path}")
    print(f"Wrote {features_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
