from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_RE = re.compile(r"\s+")


def norm_prompt(s: str) -> str:
    return PROMPT_RE.sub(" ", str(s).strip().lower())


def prompt_hash(s: str) -> str:
    return hashlib.sha1(norm_prompt(s).encode("utf-8")).hexdigest()[:16]


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


def paired_bootstrap_mean(values: np.ndarray, n_boot: int, seed: int) -> dict[str, float]:
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


def make_story_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for i, row in df.iterrows():
        for side in ["chosen", "rejected"]:
            story = row[f"{side}_story"]
            rows.append(
                {
                    "row_id": int(i),
                    "side": side,
                    "prompt": row["prompt"],
                    "prompt_hash": row["prompt_hash"],
                    "story": story,
                    "comment_id": row.get(f"{side}_comment_id", ""),
                    "reddit_post_id": row.get(f"{side}_reddit_post_id", ""),
                    "upvotes": row.get(f"{side}_upvotes", np.nan),
                    "story_hash": hashlib.sha1(str(story).encode("utf-8")).hexdigest()[:16],
                }
            )

    out = pd.DataFrame(rows)
    return out.drop_duplicates(["prompt_hash", "story_hash"]).reset_index(drop=True)


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
    elif domain_mode == "all_other":
        pass
    else:
        raise ValueError(f"Unknown domain_mode: {domain_mode}")

    if "upvotes" in group.columns:
        group = group.sort_values(["upvotes", "row_id"], ascending=[False, True])
    else:
        group = group.sort_values(["row_id"])

    return group.head(max_domain).reset_index(drop=True)


def build_prompt_only_context(prompt: str) -> str:
    return f"Prompt:\n{prompt}\n\nResponse:\n"


def build_candidate_context(prompt: str, candidate_story: str) -> str:
    return (
        f"Prompt:\n{prompt}\n\n"
        f"Response:\n{candidate_story}\n\n"
        f"Another response:\n"
    )


@torch.inference_mode()
def conditional_target_nll(
    context: str,
    target: str,
    tokenizer,
    model,
    device: str,
    max_context_tokens: int,
    max_target_tokens: int,
) -> dict[str, float]:
    context_ids = tokenizer(str(context), add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(str(target), add_special_tokens=False)["input_ids"]

    model_max = int(getattr(model.config, "n_positions", 1024))
    max_context_tokens = min(max_context_tokens, model_max - 2)
    max_target_tokens = min(max_target_tokens, model_max - 2)

    target_ids = target_ids[:max_target_tokens]

    remaining_for_context = max(1, model_max - len(target_ids))
    context_keep = min(max_context_tokens, remaining_for_context)
    context_ids = context_ids[-context_keep:]

    input_ids = context_ids + target_ids

    if len(input_ids) < 3 or len(target_ids) < 2:
        return {
            "avg_nll": float("nan"),
            "total_nll": float("nan"),
            "n_loss_tokens": 0,
            "n_context_tokens": len(context_ids),
            "n_target_tokens": len(target_ids),
        }

    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    labels = x.clone()

    n_context = len(context_ids)
    labels[:, :n_context] = -100
    if n_context == 0:
        labels[:, 0] = -100

    out = model(x, labels=labels)
    avg_nll = float(out.loss.detach().cpu())

    n_loss_tokens = len(target_ids)
    if n_context == 0:
        n_loss_tokens = max(0, n_loss_tokens - 1)

    return {
        "avg_nll": avg_nll,
        "total_nll": avg_nll * n_loss_tokens,
        "n_loss_tokens": int(n_loss_tokens),
        "n_context_tokens": int(len(context_ids)),
        "n_target_tokens": int(len(target_ids)),
    }


def score_domain(
    prompt: str,
    candidate_story: str | None,
    domain: pd.DataFrame,
    tokenizer,
    model,
    device: str,
    max_context_tokens: int,
    max_target_tokens: int,
) -> dict[str, float]:
    if candidate_story is None:
        context = build_prompt_only_context(prompt)
    else:
        context = build_candidate_context(prompt, candidate_story)

    rows = []
    for _, target in domain.iterrows():
        rows.append(
            conditional_target_nll(
                context=context,
                target=target["story"],
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_context_tokens=max_context_tokens,
                max_target_tokens=max_target_tokens,
            )
        )

    vals = pd.DataFrame(rows)
    return {
        "avg_nll": float(vals["avg_nll"].mean()),
        "total_nll": float(vals["total_nll"].sum()),
        "n_loss_tokens": int(vals["n_loss_tokens"].sum()),
        "n_domain_items": int(len(domain)),
        "mean_target_tokens": float(vals["n_target_tokens"].mean()),
        "mean_context_tokens": float(vals["n_context_tokens"].mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="SAA-Lab/LitBench-Test-IDs-Complete")
    ap.add_argument("--tag", default="test_prompt_v_distilgpt2")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--domain-mode", default="other_chosen", choices=["other_chosen", "other_rejected", "all_other"])
    ap.add_argument("--min-domain", type=int, default=2)
    ap.add_argument("--max-domain", type=int, default=4)
    ap.add_argument("--max-context-tokens", type=int, default=512)
    ap.add_argument("--max-target-tokens", type=int, default=384)
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
    df["prompt_norm"] = df["prompt"].map(norm_prompt)
    df["prompt_hash"] = df["prompt"].map(prompt_hash)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    story_table = make_story_table(df)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    rows = []
    skipped = 0

    for i, row in tqdm(df.iterrows(), total=len(df), desc="prompt-conditioned V"):
        domain = select_domain(
            story_table=story_table,
            prompt_hash_value=row["prompt_hash"],
            current_row_id=int(i),
            domain_mode=args.domain_mode,
            max_domain=args.max_domain,
        )

        if len(domain) < args.min_domain:
            skipped += 1
            continue

        prompt_only = score_domain(
            prompt=row["prompt"],
            candidate_story=None,
            domain=domain,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        chosen_cond = score_domain(
            prompt=row["prompt"],
            candidate_story=row["chosen_story"],
            domain=domain,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )
        rejected_cond = score_domain(
            prompt=row["prompt"],
            candidate_story=row["rejected_story"],
            domain=domain,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_context_tokens=args.max_context_tokens,
            max_target_tokens=args.max_target_tokens,
        )

        chosen_gain = prompt_only["avg_nll"] - chosen_cond["avg_nll"]
        rejected_gain = prompt_only["avg_nll"] - rejected_cond["avg_nll"]

        rows.append(
            {
                "row_id": int(i),
                "prompt_hash": row["prompt_hash"],
                "n_domain_items": int(len(domain)),
                "domain_mode": args.domain_mode,
                "prompt_only_avg_nll": prompt_only["avg_nll"],
                "chosen_cond_avg_nll": chosen_cond["avg_nll"],
                "rejected_cond_avg_nll": rejected_cond["avg_nll"],
                "chosen_gain": chosen_gain,
                "rejected_gain": rejected_gain,
                "v_delta_gain": chosen_gain - rejected_gain,
                "v_delta_cond_nll": rejected_cond["avg_nll"] - chosen_cond["avg_nll"],
                "chosen_wins_v": bool(chosen_gain > rejected_gain),
                "mean_target_tokens": prompt_only["mean_target_tokens"],
                "mean_prompt_context_tokens": prompt_only["mean_context_tokens"],
                "mean_candidate_context_tokens": chosen_cond["mean_context_tokens"],
                "chosen_comment_id": row.get("chosen_comment_id", ""),
                "rejected_comment_id": row.get("rejected_comment_id", ""),
                "chosen_reddit_post_id": row.get("chosen_reddit_post_id", ""),
                "rejected_reddit_post_id": row.get("rejected_reddit_post_id", ""),
            }
        )

    scores = pd.DataFrame(rows)
    if len(scores) == 0:
        raise SystemExit("No eligible rows. Lower --min-domain or check prompt grouping.")

    correct = scores["chosen_wins_v"].to_numpy(dtype=bool)
    acc = bootstrap_accuracy(correct, n_boot=args.n_boot, seed=args.seed)
    delta_stats = paired_bootstrap_mean(scores["v_delta_gain"].to_numpy(), n_boot=args.n_boot, seed=args.seed)

    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "tag": args.tag,
                "model": args.model,
                "domain_mode": args.domain_mode,
                "min_domain": args.min_domain,
                "max_domain": args.max_domain,
                "max_context_tokens": args.max_context_tokens,
                "max_target_tokens": args.max_target_tokens,
                "n_total_rows": int(len(df)),
                "n_eligible_rows": int(len(scores)),
                "n_skipped_rows": int(skipped),
                "baseline": "prefer_larger_prompt_conditioned_v",
                **acc,
                "mean_v_delta_gain": delta_stats["mean"],
                "v_delta_ci95_low": delta_stats["ci95_low"],
                "v_delta_ci95_high": delta_stats["ci95_high"],
                "p_v_delta_le_zero": delta_stats["p_le_zero"],
                "p_v_delta_ge_zero": delta_stats["p_ge_zero"],
            }
        ]
    )

    scores_path = OUT_DIR / f"litbench_prompt_conditioned_v_scores_{args.tag}.csv"
    summary_path = OUT_DIR / f"litbench_prompt_conditioned_v_summary_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_prompt_conditioned_v_manifest_{args.tag}.json"

    scores.to_csv(scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "prompt-conditioned compression-progress V",
                "dataset": args.dataset,
                "tag": args.tag,
                "model": args.model,
                "domain_mode": args.domain_mode,
                "min_domain": args.min_domain,
                "max_domain": args.max_domain,
                "max_context_tokens": args.max_context_tokens,
                "max_target_tokens": args.max_target_tokens,
                "outputs": {
                    "scores": str(scores_path),
                    "summary": str(summary_path),
                },
                "interpretation": (
                    "V(candidate) = prompt_only_domain_avg_nll - candidate_conditioned_domain_avg_nll. "
                    "Prediction is chosen if V(chosen) > V(rejected). "
                    "Domain is selected from same-prompt held-out stories excluding current pair."
                ),
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
