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
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


@torch.inference_mode()
def score_text_nll(text: str, tokenizer, model, device: str, max_length: int, stride: int) -> dict[str, float]:
    text = str(text)

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=False,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"][0]
    n_tokens_total = int(input_ids.numel())

    if n_tokens_total < 2:
        return {
            "avg_nll": float("nan"),
            "total_nll": float("nan"),
            "n_loss_tokens": 0,
            "n_tokens_total": n_tokens_total,
        }

    # GPT-style models have fixed context. Use sliding windows.
    max_length = min(max_length, getattr(model.config, "n_positions", max_length))
    stride = min(stride, max_length)

    total_nll = 0.0
    total_loss_tokens = 0
    prev_end = 0

    for begin in range(0, n_tokens_total, stride):
        end = min(begin + max_length, n_tokens_total)
        trg_len = end - prev_end
        if trg_len <= 0:
            continue

        ids = input_ids[begin:end].unsqueeze(0).to(device)
        labels = ids.clone()
        labels[:, :-trg_len] = -100

        out = model(ids, labels=labels)
        loss = float(out.loss.detach().cpu())

        # HF causal LM loss is averaged over non-ignored shifted labels.
        # Approximate valid loss tokens as trg_len, minus one if this is the first window.
        loss_tokens = trg_len
        if begin == 0:
            loss_tokens = max(0, trg_len - 1)

        if loss_tokens > 0:
            total_nll += loss * loss_tokens
            total_loss_tokens += loss_tokens

        prev_end = end
        if end == n_tokens_total:
            break

    return {
        "avg_nll": float(total_nll / total_loss_tokens) if total_loss_tokens else float("nan"),
        "total_nll": float(total_nll),
        "n_loss_tokens": int(total_loss_tokens),
        "n_tokens_total": n_tokens_total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--tag", default="test_ids_complete_distilgpt2")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install with:\n"
            "  python -m pip install transformers torch tqdm\n"
        ) from exc

    df = load_hf_dataset(args.dataset)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    required = {"chosen_story", "rejected_story"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset lacks required columns: {sorted(missing)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    rows = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="scoring pairs"):
        chosen = score_text_nll(
            row["chosen_story"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
        )
        rejected = score_text_nll(
            row["rejected_story"],
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_length=args.max_length,
            stride=args.stride,
        )

        out = {
            "row_id": i,
            "chosen_avg_nll": chosen["avg_nll"],
            "rejected_avg_nll": rejected["avg_nll"],
            "chosen_total_nll": chosen["total_nll"],
            "rejected_total_nll": rejected["total_nll"],
            "chosen_n_loss_tokens": chosen["n_loss_tokens"],
            "rejected_n_loss_tokens": rejected["n_loss_tokens"],
            "chosen_n_tokens_total": chosen["n_tokens_total"],
            "rejected_n_tokens_total": rejected["n_tokens_total"],
        }

        for optional in [
            "chosen_comment_id",
            "rejected_comment_id",
            "chosen_reddit_post_id",
            "rejected_reddit_post_id",
            "chosen_upvotes",
            "rejected_upvotes",
        ]:
            if optional in df.columns:
                out[optional] = row[optional]

        rows.append(out)

    scores = pd.DataFrame(rows)

    valid_avg = scores["chosen_avg_nll"].notna() & scores["rejected_avg_nll"].notna()
    valid_total = scores["chosen_total_nll"].notna() & scores["rejected_total_nll"].notna()

    tests = {
        "prefer_lower_avg_nll": scores.loc[valid_avg, "chosen_avg_nll"] < scores.loc[valid_avg, "rejected_avg_nll"],
        "prefer_higher_avg_nll": scores.loc[valid_avg, "chosen_avg_nll"] > scores.loc[valid_avg, "rejected_avg_nll"],
        "prefer_lower_total_nll": scores.loc[valid_total, "chosen_total_nll"] < scores.loc[valid_total, "rejected_total_nll"],
        "prefer_higher_total_nll": scores.loc[valid_total, "chosen_total_nll"] > scores.loc[valid_total, "rejected_total_nll"],
    }

    summary_rows = []
    for name, pred in tests.items():
        stat = bootstrap_accuracy(pred.to_numpy(dtype=bool), n_boot=args.n_boot, seed=args.seed)
        summary_rows.append(
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "model": args.model,
                "baseline": name,
                **stat,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("accuracy", ascending=False)

    scores_path = OUT_DIR / f"litbench_nll_scores_{args.tag}.csv"
    summary_path = OUT_DIR / f"litbench_nll_baselines_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_nll_baselines_{args.tag}_manifest.json"

    scores.to_csv(scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "model": args.model,
                "analysis": "pairwise NLL baselines",
                "n_rows": int(len(df)),
                "max_length": args.max_length,
                "stride": args.stride,
                "device": device,
                "outputs": {
                    "scores": str(scores_path),
                    "summary": str(summary_path),
                },
                "n_boot": args.n_boot,
                "seed": args.seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"Wrote {scores_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
