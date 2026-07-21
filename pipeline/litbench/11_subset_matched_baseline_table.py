from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "litbench"


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


def read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="subset1155_local_baselines_maxdomain10")
    ap.add_argument("--domain-tag", default="test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10")
    ap.add_argument("--tfidf-tag", default="test_prompt_domain_kernel_tfidf_mindomain2_maxdomain3")
    ap.add_argument("--embedding-tag", default="test_prompt_domain_kernel_tfidf_embedding_mindomain2_maxdomain3")
    ap.add_argument("--random-tag", default="test_prompt_v_distilgpt2_randomsplit_seed123_mindomain2_maxdomain3")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    domain_models = read_required(
        OUT_DIR / f"litbench_prompt_v_domain_contrast_models_{args.domain_tag}.csv"
    )
    tfidf_models = read_required(
        OUT_DIR / f"litbench_prompt_domain_kernel_models_{args.tfidf_tag}.csv"
    )
    embedding_models = read_required(
        OUT_DIR / f"litbench_prompt_domain_kernel_models_{args.embedding_tag}.csv"
    )
    random_models = read_required(
        OUT_DIR / f"litbench_prompt_v_random_split_models_{args.random_tag}.csv"
    )

    rows = []

    def add_model(source: str, df: pd.DataFrame, model_name: str, kernel: str | None = None):
        sub = df[df["model"] == model_name].copy()
        if kernel is not None and "kernel" in sub.columns:
            sub = sub[sub["kernel"] == kernel]
        if len(sub) != 1:
            raise RuntimeError(f"Expected one row for {source}/{model_name}/{kernel}, got {len(sub)}")
        r = sub.iloc[0].to_dict()
        rows.append(
            {
                "tag": args.tag,
                "source": source,
                "model": model_name,
                "kernel": kernel or r.get("kernel", ""),
                "accuracy": r["accuracy"],
                "ci95_low": r["ci95_low"],
                "ci95_high": r["ci95_high"],
                "n": int(r["n"]),
                "n_boot": int(r["n_boot"]),
            }
        )

    add_model("compression_v_maxdomain10", domain_models, "domain_contrast_sign_rule")
    add_model("compression_v_maxdomain10", domain_models, "domain_specificity_logistic")
    add_model("compression_v_maxdomain10", domain_models, "surface_format")

    add_model("random_pool_control", random_models, "random_domain_contrast_sign_rule")
    add_model("random_pool_control", random_models, "random_domain_specificity_logistic")
    add_model("random_pool_control", random_models, "surface_format")

    add_model("tfidf_kernel_control", tfidf_models, "domain_specificity_logistic", kernel="tfidf")
    add_model("tfidf_kernel_control", tfidf_models, "surface_plus_domain_specificity", kernel="tfidf")

    add_model("embedding_kernel_control", embedding_models, "domain_specificity_logistic", kernel="embedding")
    add_model("embedding_kernel_control", embedding_models, "surface_plus_domain_specificity", kernel="embedding")

    out = pd.DataFrame(rows).sort_values("accuracy", ascending=False)

    out_path = OUT_DIR / f"litbench_subset_matched_baseline_table_{args.tag}.csv"
    manifest_path = OUT_DIR / f"litbench_subset_matched_baseline_manifest_{args.tag}.json"

    out.to_csv(out_path, index=False)
    manifest_path.write_text(
        json.dumps(
            {
                "analysis": "Subset-matched local baseline table for exact repeated-prompt overlap",
                "tag": args.tag,
                "domain_tag": args.domain_tag,
                "tfidf_tag": args.tfidf_tag,
                "embedding_tag": args.embedding_tag,
                "random_tag": args.random_tag,
                "note": "This table compares already-computed local baselines. Official released LitBench verifier failed locally on CPU and should be run on GPU/HPC.",
                "outputs": {"table": str(out_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(out.to_string(index=False))
    print(f"Wrote {out_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
