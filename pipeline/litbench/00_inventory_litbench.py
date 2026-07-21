from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)


DATASETS_TO_TRY = [
    "SAA-Lab/LitBench-Test",
    "SAA-Lab/LitBench-Test-IDs-Complete",
    "SAA-Lab/LitBench-Train",
    "SAA-Lab/LitBench-Rationales",
    "SAA-Lab/LitBench",
]


def summarize_df(name: str, split: str, df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    text_like_cols = [
        c for c in cols
        if any(tok in c.lower() for tok in ["text", "story", "comment", "chosen", "rejected", "prompt", "response"])
    ]

    row = {
        "dataset_name": name,
        "split": split,
        "n_rows": int(len(df)),
        "n_cols": int(len(cols)),
        "columns": cols,
        "text_like_columns": text_like_cols,
    }

    for c in cols:
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_string_dtype(df[c]):
            nonnull = df[c].dropna().astype(str)
            if len(nonnull):
                row[f"{c}__mean_strlen"] = float(nonnull.str.len().mean())
                row[f"{c}__max_strlen"] = int(nonnull.str.len().max())
                row[f"{c}__example"] = nonnull.iloc[0][:500]

    return row


def main() -> None:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install with:\n"
            "  python -m pip install datasets pyarrow\n"
        ) from exc

    summaries = []
    sample_paths = []

    for name in DATASETS_TO_TRY:
        print(f"\n=== trying {name} ===")
        try:
            ds = load_dataset(name)
        except Exception as exc:
            print(f"FAILED {name}: {exc}")
            summaries.append(
                {
                    "dataset_name": name,
                    "status": "failed",
                    "error": repr(exc),
                }
            )
            continue

        print(ds)
        for split in ds.keys():
            df = ds[split].to_pandas()
            summary = summarize_df(name, split, df)
            summary["status"] = "loaded"
            summaries.append(summary)

            safe_name = name.replace("/", "__")
            sample_path = OUT_DIR / f"{safe_name}__{split}__sample.csv"
            df.head(50).to_csv(sample_path, index=False)
            sample_paths.append(str(sample_path))
            print(f"{name}/{split}: rows={len(df)} cols={len(df.columns)}")
            print(f"columns={list(df.columns)}")
            print(f"sample={sample_path}")

    summary_path = OUT_DIR / "litbench_inventory_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    csv_rows = []
    for s in summaries:
        csv_rows.append(
            {
                "dataset_name": s.get("dataset_name"),
                "split": s.get("split"),
                "status": s.get("status"),
                "n_rows": s.get("n_rows"),
                "n_cols": s.get("n_cols"),
                "columns": json.dumps(s.get("columns", [])),
                "text_like_columns": json.dumps(s.get("text_like_columns", [])),
                "error": s.get("error", ""),
            }
        )
    pd.DataFrame(csv_rows).to_csv(OUT_DIR / "litbench_inventory_summary.csv", index=False)

    print(f"\nWrote {summary_path}")
    print(f"Wrote {OUT_DIR / 'litbench_inventory_summary.csv'}")
    for p in sample_paths:
        print(f"Wrote {p}")


if __name__ == "__main__":
    main()
