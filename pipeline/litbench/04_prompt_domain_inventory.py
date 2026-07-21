from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS = {
    "test_ids_complete": "SAA-Lab/LitBench-Test-IDs-Complete",
    "train": "SAA-Lab/LitBench-Train",
    "rationales": "SAA-Lab/LitBench-Rationales",
}


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


def norm_prompt(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def prompt_hash(s: str) -> str:
    return hashlib.sha1(norm_prompt(s).encode("utf-8")).hexdigest()[:16]


def add_prompt_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "prompt" in out.columns:
        out["prompt_norm"] = out["prompt"].map(norm_prompt)
        out["prompt_hash"] = out["prompt"].map(prompt_hash)
    return out


def count_distribution(series: pd.Series) -> dict:
    counts = series.value_counts(dropna=False)
    return {
        "n_unique": int(counts.shape[0]),
        "n_total": int(series.shape[0]),
        "max_count": int(counts.max()) if len(counts) else 0,
        "n_values_count_ge_2": int((counts >= 2).sum()),
        "n_values_count_ge_3": int((counts >= 3).sum()),
        "n_values_count_ge_5": int((counts >= 5).sum()),
        "n_values_count_ge_10": int((counts >= 10).sum()),
        "top_counts": counts.head(20).to_dict(),
    }


def story_long(df: pd.DataFrame, dataset_key: str) -> pd.DataFrame:
    rows = []

    if {"chosen_story", "rejected_story"} <= set(df.columns):
        for side in ["chosen", "rejected"]:
            story_col = f"{side}_story"
            item = pd.DataFrame(
                {
                    "dataset_key": dataset_key,
                    "side": side,
                    "prompt": df["prompt"] if "prompt" in df.columns else "",
                    "prompt_norm": df["prompt_norm"] if "prompt_norm" in df.columns else "",
                    "prompt_hash": df["prompt_hash"] if "prompt_hash" in df.columns else "",
                    "story": df[story_col],
                }
            )
            if f"{side}_comment_id" in df.columns:
                item["comment_id"] = df[f"{side}_comment_id"]
            if f"{side}_reddit_post_id" in df.columns:
                item["reddit_post_id"] = df[f"{side}_reddit_post_id"]
            if f"{side}_upvotes" in df.columns:
                item["upvotes"] = df[f"{side}_upvotes"]
            rows.append(item)

    elif {"story_a", "story_b"} <= set(df.columns):
        for side in ["a", "b"]:
            story_col = f"story_{side}"
            item = pd.DataFrame(
                {
                    "dataset_key": dataset_key,
                    "side": side,
                    "prompt": df["prompt"] if "prompt" in df.columns else "",
                    "prompt_norm": df["prompt_norm"] if "prompt_norm" in df.columns else "",
                    "prompt_hash": df["prompt_hash"] if "prompt_hash" in df.columns else "",
                    "story": df[story_col],
                    "chosen_story_label": df["chosen_story"] if "chosen_story" in df.columns else "",
                }
            )
            rows.append(item)

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def main() -> None:
    frames = {}
    story_frames = {}

    for key, name in DATASETS.items():
        print(f"Loading {key}: {name}")
        df = add_prompt_keys(load_hf_dataset(name))
        frames[key] = df

        out_path = OUT_DIR / f"litbench_{key}_prompt_keys_sample.csv"
        df.head(100).to_csv(out_path, index=False)
        print(f"Wrote {out_path}")

        long = story_long(df, key)
        story_frames[key] = long

    summary_rows = []

    for key, df in frames.items():
        row = {
            "dataset_key": key,
            "n_rows": int(len(df)),
            "columns": json.dumps(list(df.columns)),
        }

        if "prompt_hash" in df.columns:
            dist = count_distribution(df["prompt_hash"])
            for k, v in dist.items():
                if k != "top_counts":
                    row[f"prompt_{k}"] = v
            row["prompt_top_counts_json"] = json.dumps(dist["top_counts"])

        if "chosen_reddit_post_id" in df.columns:
            dist = count_distribution(df["chosen_reddit_post_id"])
            for k, v in dist.items():
                if k != "top_counts":
                    row[f"chosen_post_{k}"] = v
            row["chosen_post_top_counts_json"] = json.dumps(dist["top_counts"])

        if "rejected_reddit_post_id" in df.columns:
            dist = count_distribution(df["rejected_reddit_post_id"])
            for k, v in dist.items():
                if k != "top_counts":
                    row[f"rejected_post_{k}"] = v
            row["rejected_post_top_counts_json"] = json.dumps(dist["top_counts"])

        summary_rows.append(row)

    # Prompt overlaps.
    overlap_rows = []
    keys = list(frames.keys())
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if "prompt_hash" not in frames[a].columns or "prompt_hash" not in frames[b].columns:
                continue
            set_a = set(frames[a]["prompt_hash"])
            set_b = set(frames[b]["prompt_hash"])
            inter = set_a & set_b
            overlap_rows.append(
                {
                    "dataset_a": a,
                    "dataset_b": b,
                    "n_prompt_a": len(set_a),
                    "n_prompt_b": len(set_b),
                    "n_prompt_overlap": len(inter),
                    "frac_a_overlap": len(inter) / len(set_a) if set_a else 0.0,
                    "frac_b_overlap": len(inter) / len(set_b) if set_b else 0.0,
                }
            )

    # Long story-level domain inventory.
    all_stories = pd.concat(
        [x for x in story_frames.values() if len(x)],
        ignore_index=True,
    )
    all_stories["story_chars"] = all_stories["story"].astype(str).str.len()
    all_stories["story_words"] = all_stories["story"].astype(str).str.split().str.len()

    story_summary_rows = []
    for key, group in all_stories.groupby("dataset_key"):
        prompt_counts = group["prompt_hash"].value_counts()
        story_summary_rows.append(
            {
                "dataset_key": key,
                "n_story_rows": int(len(group)),
                "n_unique_prompts": int(group["prompt_hash"].nunique()),
                "max_stories_per_prompt": int(prompt_counts.max()) if len(prompt_counts) else 0,
                "n_prompts_ge_4_stories": int((prompt_counts >= 4).sum()),
                "n_prompts_ge_6_stories": int((prompt_counts >= 6).sum()),
                "n_prompts_ge_10_stories": int((prompt_counts >= 10).sum()),
                "mean_story_words": float(group["story_words"].mean()),
                "median_story_words": float(group["story_words"].median()),
                "mean_story_chars": float(group["story_chars"].mean()),
                "median_story_chars": float(group["story_chars"].median()),
            }
        )

    summary = pd.DataFrame(summary_rows)
    overlaps = pd.DataFrame(overlap_rows)
    story_summary = pd.DataFrame(story_summary_rows)

    summary_path = OUT_DIR / "litbench_prompt_domain_inventory.csv"
    overlaps_path = OUT_DIR / "litbench_prompt_overlap_inventory.csv"
    story_summary_path = OUT_DIR / "litbench_story_domain_inventory.csv"
    all_stories_sample_path = OUT_DIR / "litbench_story_long_sample.csv"
    manifest_path = OUT_DIR / "litbench_prompt_domain_inventory_manifest.json"

    summary.to_csv(summary_path, index=False)
    overlaps.to_csv(overlaps_path, index=False)
    story_summary.to_csv(story_summary_path, index=False)
    all_stories.head(500).to_csv(all_stories_sample_path, index=False)

    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "LitBench prompt/domain feasibility inventory",
                "datasets": DATASETS,
                "outputs": {
                    "summary": str(summary_path),
                    "overlaps": str(overlaps_path),
                    "story_summary": str(story_summary_path),
                    "story_long_sample": str(all_stories_sample_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== Prompt/domain summary ===")
    print(summary.to_string(index=False))
    print("\n=== Prompt overlaps ===")
    print(overlaps.to_string(index=False))
    print("\n=== Story-level domain summary ===")
    print(story_summary.to_string(index=False))

    print(f"\nWrote {summary_path}")
    print(f"Wrote {overlaps_path}")
    print(f"Wrote {story_summary_path}")
    print(f"Wrote {all_stories_sample_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
