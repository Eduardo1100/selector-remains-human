from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"


def load_hf_dataset(name: str) -> pd.DataFrame:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets with: python -m pip install datasets pyarrow") from exc

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


def load_overlap_rows(domain_tag: str) -> pd.DataFrame:
    path = OUT_DIR / f"litbench_prompt_v_domain_contrast_scores_{domain_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_surface(surface_tag: str) -> pd.DataFrame:
    path = OUT_DIR / f"litbench_surface_features_{surface_tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def score_texts_reward_model(
    texts: list[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: str,
    dtype_name: str,
) -> np.ndarray:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[dtype_name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        num_labels=1,
    )
    model.to(device)
    model.eval()

    scores = []

    for start in tqdm(range(0, len(texts), batch_size), desc="reward scoring"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.squeeze(-1).detach().float().cpu().numpy()

        scores.extend(logits.tolist())

    return np.asarray(scores, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--surface-tag", default="test_ids_complete")
    ap.add_argument("--domain-tag", default="test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10")
    ap.add_argument("--tag", default="subset1155_litbench_reward_baseline_maxdomain10")
    ap.add_argument("--reward-model", default="SAA-Lab/Llama8B-CreativeWritingVerifier")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16" if torch.cuda.is_available() else "float32")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    df = load_hf_dataset(args.dataset)
    overlap = load_overlap_rows(args.domain_tag)
    surface = load_surface(args.surface_tag)

    row_ids = overlap["row_id"].astype(int).tolist()
    sub = df.iloc[row_ids].copy()
    sub.insert(0, "row_id", row_ids)

    surface_sub = surface[surface["row_id"].isin(row_ids)].copy()
    surface_sub = surface_sub.sort_values("row_id")

    if len(surface_sub) != len(sub):
        raise RuntimeError(f"Surface subset mismatch: {len(surface_sub)} vs {len(sub)}")

    # Surface baseline using the existing pairwise model prediction if available is not stored here,
    # so this script reports simple direct surface rules plus reward-model score.
    direct_rules = {}

    direct_rules["prefer_more_paragraphs"] = (
        surface_sub["chosen_paragraph_count"].to_numpy() > surface_sub["rejected_paragraph_count"].to_numpy()
    )
    direct_rules["prefer_more_newlines"] = (
        surface_sub["chosen_newline_count"].to_numpy() > surface_sub["rejected_newline_count"].to_numpy()
    )
    direct_rules["prefer_more_punct"] = (
        surface_sub["chosen_punct_count"].to_numpy() > surface_sub["rejected_punct_count"].to_numpy()
    )
    direct_rules["prefer_more_words"] = (
        surface_sub["chosen_words"].to_numpy() > surface_sub["rejected_words"].to_numpy()
    )

    print(f"Scoring reward model {args.reward_model} on {len(sub)} pairs using device={args.device}")

    chosen_scores = score_texts_reward_model(
        sub["chosen_story"].astype(str).tolist(),
        model_name=args.reward_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        dtype_name=args.dtype,
    )
    rejected_scores = score_texts_reward_model(
        sub["rejected_story"].astype(str).tolist(),
        model_name=args.reward_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        dtype_name=args.dtype,
    )

    reward_correct = chosen_scores > rejected_scores
    direct_rules["reward_model"] = reward_correct

    rows = []
    correctness = {}

    for name, correct in direct_rules.items():
        correct = np.asarray(correct, dtype=bool)
        correctness[name] = correct
        rows.append(
            {
                "tag": args.tag,
                "model": name,
                **bootstrap_accuracy(correct, n_boot=args.n_boot, seed=args.seed),
            }
        )

    # Compare reward model to simple direct rules.
    delta_rows = []
    for other in ["prefer_more_paragraphs", "prefer_more_newlines", "prefer_more_punct", "prefer_more_words"]:
        delta_rows.append(
            {
                "tag": args.tag,
                "model_a": "reward_model",
                "model_b": other,
                **paired_bootstrap_delta(correctness["reward_model"], correctness[other], args.n_boot, args.seed),
            }
        )

    score_df = sub[["row_id", "prompt", "chosen_story", "rejected_story"]].copy()
    score_df["chosen_reward_score"] = chosen_scores
    score_df["rejected_reward_score"] = rejected_scores
    score_df["reward_delta"] = chosen_scores - rejected_scores
    score_df["reward_correct"] = reward_correct

    models = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
    deltas = pd.DataFrame(delta_rows)

    scores_path = OUT_DIR / f"litbench_subset_reward_scores_{args.tag}.csv"
    models_path = OUT_DIR / f"litbench_subset_reward_models_{args.tag}.csv"
    deltas_path = OUT_DIR / f"litbench_subset_reward_deltas_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_subset_reward_manifest_{args.tag}.json"

    score_df.to_csv(scores_path, index=False)
    models.to_csv(models_path, index=False)
    deltas.to_csv(deltas_path, index=False)

    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "Subset-matched reward model baseline on exact domain-contrast overlap",
                "dataset": args.dataset,
                "surface_tag": args.surface_tag,
                "domain_tag": args.domain_tag,
                "tag": args.tag,
                "reward_model": args.reward_model,
                "n_rows": int(len(sub)),
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "device": args.device,
                "dtype": args.dtype,
                "seed": args.seed,
                "n_boot": args.n_boot,
                "outputs": {
                    "scores": str(scores_path),
                    "models": str(models_path),
                    "deltas": str(deltas_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(models.to_string(index=False))
    print()
    print(deltas.to_string(index=False))
    print(f"Wrote {scores_path}")
    print(f"Wrote {models_path}")
    print(f"Wrote {deltas_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
