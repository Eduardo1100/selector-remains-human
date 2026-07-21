from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import re
import tempfile
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
SCORES = ROOT / "results" / "scores" / "phase_a_eval_scores"
PHASE_A_RESULTS = ROOT / "results" / "phase_a_eval_results"
OUT = ROOT / "results" / "analyses"
HASH_OUT = ROOT / "results" / "hashes"

HUMAN_CONTROLS = [
    "target__Aesthetic_Appeal",
    "target__Clarity",
    "target__Creativity",
    "target__Felt_Valence",
]

SURFACE_CONTROLS = [
    "word_len_calc",
    "char_len_calc",
    "line_count",
]

NLL_CONTROLS = [
    "item_nll_bpb__distilgpt2",
    "item_nll_bpb__gpt2",
    "item_nll_bpb__gpt2-medium",
]

FEATURE_SETS = {
    "none": [],
    "other_human_targets": HUMAN_CONTROLS,
    "other_human_plus_surface": HUMAN_CONTROLS + SURFACE_CONTROLS,
    "stacked": HUMAN_CONTROLS + SURFACE_CONTROLS + NLL_CONTROLS,
}

DEFAULT_METRICS = [
    "score_pref_struct",
]

ALL_METRICS = [
    "score_pref_struct",
    "score_pref_raw",
]

DEFAULT_FEATURE_SET_NAMES = [
    "other_human_targets",
    "other_human_plus_surface",
    "stacked",
]

BASELINE_FILES = [
    SCORES / "supervised_similarity_baselines_tfidf_kfold_surface_chaudhuri_Surprise.csv",
    SCORES / "supervised_similarity_baselines_embedding_kfold_surface_chaudhuri_Surprise.csv",
]

BASELINE_METHOD_MAP = {
    "tfidf": "tfidf_contrast",
    "embedding": "embedding_contrast",
}

COMPRESSION_METHOD = "compression_distilgpt2"
PARITY_DRAWS = 20
PARITY_TOLERANCE = 1e-10
CHECKPOINT_SAMPLES = OUT / ".bootstrap_absolute_effects_samples.checkpoint.csv"
CHECKPOINT_STATE = OUT / ".bootstrap_absolute_effects_checkpoint.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def residualize_rank(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y_rank = rankdata(y)

    if X.shape[1] == 0:
        return y_rank

    X_rank = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    model = LinearRegression()
    model.fit(X_rank, y_rank)
    return y_rank - model.predict(X_rank)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 5:
        return np.nan
    if np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return np.nan

    rho, _ = spearmanr(x, y)
    return float(rho) if np.isfinite(rho) else np.nan


def canonicalize_after_merge(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["target__Surprise"] + HUMAN_CONTROLS:
        if col in df.columns:
            continue

        for alt in [f"{col}_base", f"{col}_rating", f"{col}_x", f"{col}_y"]:
            if alt in df.columns:
                df[col] = df[alt]
                break

    if "target__Surprise" not in df.columns and "target_value" in df.columns:
        df["target__Surprise"] = df["target_value"]

    aliases = {
        "v_pref_raw": ["score_pref_raw"],
        "v_pref_struct": ["score_pref_struct"],
        "v_pref_ctrl": ["score_pref_ctrl"],
        "score_pref_raw": ["v_pref_raw"],
        "score_pref_struct": ["v_pref_struct"],
        "score_pref_ctrl": ["v_pref_ctrl"],
    }

    for canonical, alts in aliases.items():
        if canonical in df.columns:
            continue
        for alt in alts:
            if alt in df.columns:
                df[canonical] = df[alt]
                break

    return df


def load_base() -> pd.DataFrame:
    base = pd.read_csv(DATA / "targets_wide.csv")
    base = base[base["dataset"] == "chaudhuri_2024"].copy()

    keep = [
        "item_id",
        "target__Surprise",
        "target__Aesthetic_Appeal",
        "target__Clarity",
        "target__Creativity",
        "target__Felt_Valence",
    ]
    base = base[keep].copy()

    for obs in ["distilgpt2", "gpt2", "gpt2-medium"]:
        path = PHASE_A_RESULTS / f"item_unconditional_nll_{obs}.csv"
        if not path.exists():
            raise SystemExit(f"Missing NLL file: {path}")

        nll = pd.read_csv(path)
        col = f"item_nll_bpb__{obs}"

        if col not in nll.columns:
            candidates = [c for c in nll.columns if "nll" in c.lower() and "bpb" in c.lower()]
            if len(candidates) == 1:
                nll = nll.rename(columns={candidates[0]: col})
            else:
                raise SystemExit(
                    f"Could not identify NLL column for {obs} in {path}. "
                    f"Columns: {nll.columns.tolist()}"
                )

        base = base.merge(nll[["item_id", col]], on="item_id", how="left")

    return base.sort_values("item_id").reset_index(drop=True)


def load_compression(base: pd.DataFrame) -> pd.DataFrame:
    paths = sorted(
        p for p in SCORES.glob(
            "vscore_distilgpt2_prefcontrast_kfold_surface_chaudhuri_Surprise_foldseed*_seed*_dn8.csv"
        )
        if not p.name.endswith(".correlations.csv")
    )

    rows = []
    for path in paths:
        m = re.search(r"foldseed(\d+)_seed(\d+)_dn8\.csv$", path.name)
        if not m:
            continue

        df = pd.read_csv(path)
        df["fold_seed"] = int(m.group(1))
        df["seed"] = int(m.group(2))
        df["method"] = COMPRESSION_METHOD
        df = df.merge(base, on="item_id", how="left", suffixes=("", "_base"))
        df = canonicalize_after_merge(df)
        rows.append(df)

    if not rows:
        raise SystemExit("No compression score files found.")

    out = pd.concat(rows, ignore_index=True)
    require_columns(out, "compression")
    return out


def load_baselines(base: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for path in BASELINE_FILES:
        if not path.exists():
            raise SystemExit(f"Missing baseline file: {path}")

        df = pd.read_csv(path)
        df["method"] = df["method"].map(BASELINE_METHOD_MAP).fillna(df["method"])
        df = df.merge(base, on="item_id", how="left", suffixes=("", "_base"))
        df = canonicalize_after_merge(df)
        rows.append(df)

    out = pd.concat(rows, ignore_index=True)
    require_columns(out, "baselines")
    return out


def require_columns(df: pd.DataFrame, label: str) -> None:
    required = set(
        [
            "item_id",
            "fold_seed",
            "seed",
            "method",
            "target__Surprise",
            "score_pref_raw",
            "score_pref_struct",
        ]
        + HUMAN_CONTROLS
        + SURFACE_CONTROLS
        + NLL_CONTROLS
    )
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise SystemExit(f"{label} missing required columns: {missing}")


def resample_run(run: pd.DataFrame, sample_item_ids: list[str]) -> pd.DataFrame:
    indexed = run.set_index("item_id", drop=False)
    return indexed.loc[sample_item_ids].reset_index(drop=True)


def metric_rho(run: pd.DataFrame, metric_col: str, feature_controls: list[str]) -> float:
    needed = ["target__Surprise", metric_col] + feature_controls
    run = run.dropna(subset=needed).copy()

    if len(run) < 5:
        return np.nan

    if feature_controls:
        X = run[feature_controls].to_numpy(dtype=float)
    else:
        X = np.zeros((len(run), 0), dtype=float)

    y_resid = residualize_rank(run["target__Surprise"].to_numpy(dtype=float), X)
    m_resid = residualize_rank(run[metric_col].to_numpy(dtype=float), X)

    return safe_spearman(m_resid, y_resid)


def observed_absolute_effects(
    all_scores: pd.DataFrame,
    feature_set_names: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    rows = []

    selected_feature_sets = {name: FEATURE_SETS[name] for name in feature_set_names}

    for method, mdf in all_scores.groupby("method"):
        runs = {
            key: run.sort_values("item_id").reset_index(drop=True)
            for key, run in mdf.groupby(["fold_seed", "seed"])
        }

        for feature_set, controls in selected_feature_sets.items():
            for metric in metrics:
                rhos = []
                for run in runs.values():
                    rho = metric_rho(run, metric, controls)
                    if np.isfinite(rho):
                        rhos.append(rho)

                rows.append({
                    "analysis": "absolute_effect",
                    "method": method,
                    "feature_set": feature_set,
                    "metric": metric,
                    "n_runs": len(rhos),
                    "observed_mean_rho": float(np.mean(rhos)) if rhos else np.nan,
                })

    return pd.DataFrame(rows)


def bootstrap_absolute_effects_legacy(
    all_scores: pd.DataFrame,
    draws: list[tuple[list[str], np.ndarray]],
    feature_set_names: list[str],
    metrics: list[str],
    progress_label: str = "legacy absolute bootstrap",
) -> pd.DataFrame:
    records = []

    run_maps = {
        method: {
            key: run.sort_values("item_id").reset_index(drop=True)
            for key, run in mdf.groupby(["fold_seed", "seed"])
        }
        for method, mdf in all_scores.groupby("method")
    }

    selected_feature_sets = {name: FEATURE_SETS[name] for name in feature_set_names}

    for b, (sample_ids, _) in enumerate(draws):

        for method, runs in run_maps.items():
            for feature_set, controls in selected_feature_sets.items():
                for metric in metrics:
                    rhos = []

                    for run in runs.values():
                        brun = resample_run(run, sample_ids)
                        rho = metric_rho(brun, metric, controls)
                        if np.isfinite(rho):
                            rhos.append(rho)

                    records.append({
                        "bootstrap_id": b,
                        "analysis": "absolute_effect",
                        "method": method,
                        "feature_set": feature_set,
                        "metric": metric,
                        "boot_mean_rho": float(np.mean(rhos)) if rhos else np.nan,
                        "n_runs": len(rhos),
                    })

        if (b + 1) % max(1, len(draws) // 10) == 0:
            print(f"  {progress_label} {b + 1}/{len(draws)}", flush=True)

    return pd.DataFrame(records)


@dataclass
class IndexedBootstrapInputs:
    item_ids: list[str]
    methods: list[str]
    run_keys: dict[str, list[tuple[int, int]]]
    target: np.ndarray
    controls: dict[str, np.ndarray]
    scores: dict[str, dict[str, np.ndarray]]


def generate_bootstrap_draws(
    item_ids: list[str], n_boot: int, seed: int
) -> list[tuple[list[str], np.ndarray]]:
    """Match the legacy seeded Generator.choice(item_ids, ...) draws exactly."""
    rng = np.random.default_rng(seed)
    canonical = list(item_ids)
    index_by_id = {item_id: index for index, item_id in enumerate(canonical)}
    draws = []
    for _ in range(n_boot):
        sample_ids = rng.choice(canonical, size=len(canonical), replace=True).tolist()
        sample_indices = np.fromiter(
            (index_by_id[item_id] for item_id in sample_ids), dtype=np.intp, count=len(sample_ids)
        )
        draws.append((sample_ids, sample_indices))
    return draws


def preindex_bootstrap_inputs(
    base: pd.DataFrame,
    all_scores: pd.DataFrame,
    metrics: list[str],
) -> IndexedBootstrapInputs:
    """Materialize canonical item-aligned arrays once, outside bootstrap loops."""
    item_ids = base["item_id"].tolist()
    canonical = base.set_index("item_id", drop=False)
    required_controls = sorted({control for controls in FEATURE_SETS.values() for control in controls})
    feature_rows = (
        all_scores.drop_duplicates("item_id", keep="first")
        .set_index("item_id", drop=False)
        .loc[item_ids]
    )
    target = canonical["target__Surprise"].to_numpy(dtype=float)
    controls = {
        feature_set: feature_rows[columns].to_numpy(dtype=float)
        if columns
        else np.empty((len(canonical), 0), dtype=float)
        for feature_set, columns in FEATURE_SETS.items()
    }

    methods = []
    run_keys: dict[str, list[tuple[int, int]]] = {}
    scores = {metric: {} for metric in metrics}
    for method, method_df in all_scores.groupby("method"):
        methods.append(method)
        runs = []
        for key, run in method_df.groupby(["fold_seed", "seed"]):
            indexed = run.set_index("item_id", drop=False)
            aligned = indexed.loc[item_ids].reset_index(drop=True)
            if len(aligned) != len(item_ids) or aligned["item_id"].tolist() != item_ids:
                raise ValueError(
                    f"{method} {key} does not have exactly one row for each canonical poem; "
                    "use the legacy engine for this input."
                )
            runs.append((key, aligned))

        run_keys[method] = [key for key, _ in runs]
        for metric in metrics:
            scores[metric][method] = np.stack(
                [run[metric].to_numpy(dtype=float) for _, run in runs], axis=0
            )

    # This forces a fail-fast error if a future feature set references a noncanonical column.
    assert set(required_controls).issubset(feature_rows.columns)
    return IndexedBootstrapInputs(item_ids, methods, run_keys, target, controls, scores)


def batched_rank_residual_spearman(
    target: np.ndarray,
    controls: np.ndarray,
    score_vectors: np.ndarray,
) -> np.ndarray:
    """Compute legacy rank-residual Spearman correlations for many score rows at once."""
    base_valid = np.isfinite(target)
    if controls.shape[1]:
        base_valid &= np.isfinite(controls).all(axis=1)

    result = np.full(score_vectors.shape[0], np.nan, dtype=float)
    groups: dict[bytes, tuple[np.ndarray, list[int]]] = {}
    for index, score in enumerate(score_vectors):
        mask = base_valid & np.isfinite(score)
        key = mask.tobytes()
        if key not in groups:
            groups[key] = (mask, [])
        groups[key][1].append(index)

    for mask, indices in groups.values():
        if int(mask.sum()) < 5:
            continue
        y = target[mask]
        score_matrix = score_vectors[indices][:, mask]
        ranked = rankdata(np.vstack([y, score_matrix]), axis=1)

        if controls.shape[1]:
            ranked_controls = rankdata(controls[mask], axis=0)
            design = np.column_stack([np.ones(len(y)), ranked_controls])
            coefficients = np.linalg.lstsq(design, ranked.T, rcond=None)[0]
            residuals = ranked - (design @ coefficients).T
        else:
            residuals = ranked

        y_residual = residuals[0]
        score_residuals = residuals[1:]
        if np.std(y_residual) == 0:
            continue

        reranked = rankdata(np.vstack([y_residual, score_residuals]), axis=1)
        y_ranked = reranked[0] - reranked[0].mean()
        score_ranked = reranked[1:] - reranked[1:].mean(axis=1, keepdims=True)
        denominator = np.sqrt(np.sum(score_ranked**2, axis=1) * np.sum(y_ranked**2))
        nonconstant = denominator != 0
        correlations = np.full(len(indices), np.nan, dtype=float)
        correlations[nonconstant] = (
            score_ranked[nonconstant] @ y_ranked
        ) / denominator[nonconstant]
        result[indices] = correlations

    return result


def optimized_rhos_for_draw(
    indexed: IndexedBootstrapInputs,
    sample_indices: np.ndarray,
    feature_set_names: list[str],
    metrics: list[str],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    target = indexed.target[sample_indices]
    rhos: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for feature_set in feature_set_names:
        controls = indexed.controls[feature_set][sample_indices]
        for metric in metrics:
            blocks = [
                indexed.scores[metric][method][:, sample_indices] for method in indexed.methods
            ]
            combined = batched_rank_residual_spearman(target, controls, np.vstack(blocks))
            offset = 0
            rhos[(feature_set, metric)] = {}
            for method, block in zip(indexed.methods, blocks, strict=True):
                next_offset = offset + len(block)
                rhos[(feature_set, metric)][method] = combined[offset:next_offset]
                offset = next_offset
    return rhos


def bootstrap_absolute_effects_optimized(
    indexed: IndexedBootstrapInputs,
    draws: list[tuple[list[str], np.ndarray]],
    feature_set_names: list[str],
    metrics: list[str],
    records: list[dict] | None = None,
    checkpoint_every: int = 0,
    checkpoint_callback=None,
) -> pd.DataFrame:
    """Vectorized bootstrap engine with the legacy record layout and aggregation."""
    records = [] if records is None else records
    records_per_boot = len(indexed.methods) * len(feature_set_names) * len(metrics)
    if len(records) % records_per_boot:
        raise ValueError("Checkpoint row count is not aligned to complete bootstrap draws.")
    start_boot = len(records) // records_per_boot

    for b in range(start_boot, len(draws)):
        _, sample_indices = draws[b]
        rhos = optimized_rhos_for_draw(indexed, sample_indices, feature_set_names, metrics)
        for method in indexed.methods:
            for feature_set in feature_set_names:
                for metric in metrics:
                    method_rhos = rhos[(feature_set, metric)][method]
                    finite = method_rhos[np.isfinite(method_rhos)]
                    records.append({
                        "bootstrap_id": b,
                        "analysis": "absolute_effect",
                        "method": method,
                        "feature_set": feature_set,
                        "metric": metric,
                        "boot_mean_rho": float(np.mean(finite)) if len(finite) else np.nan,
                        "n_runs": len(finite),
                    })

        completed = b + 1
        if completed % max(1, len(draws) // 10) == 0:
            print(f"  optimized absolute bootstrap {completed}/{len(draws)}", flush=True)
        if checkpoint_callback and checkpoint_every and completed % checkpoint_every == 0:
            checkpoint_callback(records, completed)

    if checkpoint_callback:
        checkpoint_callback(records, len(draws))
    return pd.DataFrame(records)


def legacy_rhos_for_draw(
    all_scores: pd.DataFrame,
    sample_ids: list[str],
    feature_set_names: list[str],
    metrics: list[str],
) -> dict[tuple[str, str], dict[str, np.ndarray]]:
    """Reference run-level values used by the deterministic parity test."""
    result: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for method, method_df in all_scores.groupby("method"):
        runs = [
            run.sort_values("item_id").reset_index(drop=True)
            for _, run in method_df.groupby(["fold_seed", "seed"])
        ]
        for feature_set in feature_set_names:
            for metric in metrics:
                result.setdefault((feature_set, metric), {})[method] = np.array([
                    metric_rho(resample_run(run, sample_ids), metric, FEATURE_SETS[feature_set])
                    for run in runs
                ])
    return result


def run_parity_check(
    all_scores: pd.DataFrame,
    indexed: IndexedBootstrapInputs,
    feature_set_names: list[str],
    metrics: list[str],
    seed: int,
    n_draws: int = PARITY_DRAWS,
    tolerance: float = PARITY_TOLERANCE,
) -> float:
    """Compare every method, feature set, metric, and run before full execution."""
    maximum = 0.0
    for draw_number, (sample_ids, sample_indices) in enumerate(
        generate_bootstrap_draws(indexed.item_ids, n_draws, seed), start=1
    ):
        legacy = legacy_rhos_for_draw(all_scores, sample_ids, feature_set_names, metrics)
        optimized = optimized_rhos_for_draw(indexed, sample_indices, feature_set_names, metrics)
        for key, by_method in legacy.items():
            for method, reference in by_method.items():
                candidate = optimized[key][method]
                if not np.array_equal(np.isnan(reference), np.isnan(candidate)):
                    raise AssertionError(f"Parity NaN mismatch at draw {draw_number}, {key}, {method}.")
                difference = np.nanmax(np.abs(reference - candidate), initial=0.0)
                maximum = max(maximum, float(difference))
        print(f"  parity draw {draw_number}/{n_draws}", flush=True)

    print(
        f"Parity check: {n_draws} draws, maximum absolute run-level difference {maximum:.3e} "
        f"(tolerance {tolerance:.1e})",
        flush=True,
    )
    if maximum > tolerance:
        raise AssertionError(
            f"Optimized bootstrap parity failed: maximum difference {maximum:.3e} exceeds {tolerance:.1e}."
        )
    return maximum


def summarize_absolute(observed: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    key_cols = ["analysis", "method", "feature_set", "metric"]

    for key, obs_row in observed.groupby(key_cols):
        analysis, method, feature_set, metric = key
        sub = boot[
            (boot["analysis"] == analysis)
            & (boot["method"] == method)
            & (boot["feature_set"] == feature_set)
            & (boot["metric"] == metric)
        ]["boot_mean_rho"].dropna().to_numpy(dtype=float)

        if len(sub) == 0:
            continue

        obs = float(obs_row["observed_mean_rho"].iloc[0])
        ci_low, ci_high = np.percentile(sub, [2.5, 97.5])

        rows.append({
            "analysis": analysis,
            "method": method,
            "feature_set": feature_set,
            "metric": metric,
            "n_runs": int(obs_row["n_runs"].iloc[0]),
            "observed_mean_rho": obs,
            "boot_mean": float(np.mean(sub)),
            "boot_std": float(np.std(sub, ddof=1)),
            "ci95_low": float(ci_low),
            "ci95_high": float(ci_high),
            "p_boot_le_zero": float(np.mean(sub <= 0)),
            "p_boot_ge_zero": float(np.mean(sub >= 0)),
            "n_boot": len(sub),
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        })

    return pd.DataFrame(rows)


def write_hashes(paths: list[Path]) -> None:
    HASH_OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in paths:
        df = pd.read_csv(path) if path.suffix == ".csv" else None
        rows.append({
            "path": str(path.relative_to(ROOT)),
            "rows": len(df) if df is not None else None,
            "cols": len(df.columns) if df is not None else None,
            "sha256": sha256_file(path),
        })
    pd.DataFrame(rows).to_csv(HASH_OUT / "bootstrap_absolute_effects_hashes.csv", index=False)


def atomic_write_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        df.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".tmp", prefix=f".{path.stem}.", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2))
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def checkpoint_callback_factory(
    n_boot: int,
    seed: int,
    feature_set_names: list[str],
    metrics: list[str],
    checkpoint_samples: Path = CHECKPOINT_SAMPLES,
    checkpoint_state: Path = CHECKPOINT_STATE,
):
    def write_checkpoint(records: list[dict], completed: int) -> None:
        atomic_write_dataframe(pd.DataFrame(records), checkpoint_samples)
        atomic_write_json(
            {
                "analysis": "absolute_effect",
                "n_boot": n_boot,
                "seed": seed,
                "feature_set_names": feature_set_names,
                "metrics": metrics,
                "completed_bootstraps": completed,
            },
            checkpoint_state,
        )
        print(f"  checkpointed {completed}/{n_boot} bootstraps", flush=True)

    return write_checkpoint


def run_checkpoint_smoke_test() -> None:
    """Exercise atomic CSV/JSON writes and an optimized checkpoint boundary in isolation."""
    with tempfile.TemporaryDirectory(prefix="poemforge-stage61-checkpoint-") as directory:
        temporary_dir = Path(directory)
        csv_path = temporary_dir / "checkpoint.csv"
        json_path = temporary_dir / "checkpoint.json"

        atomic_write_dataframe(pd.DataFrame([{"value": 1}]), csv_path)
        atomic_write_json({"kind": "smoke"}, json_path)
        assert pd.read_csv(csv_path).to_dict(orient="records") == [{"value": 1}]
        assert json.loads(json_path.read_text(encoding="utf-8")) == {"kind": "smoke"}

        item_ids = ["a", "b", "c", "d", "e"]
        indexed = IndexedBootstrapInputs(
            item_ids=item_ids,
            methods=["smoke_method"],
            run_keys={"smoke_method": [(0, 0)]},
            target=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            controls={"none": np.empty((5, 0), dtype=float)},
            scores={
                "score_pref_struct": {
                    "smoke_method": np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
                }
            },
        )
        draws = [(item_ids, np.arange(len(item_ids), dtype=np.intp))]
        callback = checkpoint_callback_factory(
            1,
            123,
            ["none"],
            ["score_pref_struct"],
            checkpoint_samples=csv_path,
            checkpoint_state=json_path,
        )
        boot = bootstrap_absolute_effects_optimized(
            indexed,
            draws,
            ["none"],
            ["score_pref_struct"],
            checkpoint_every=1,
            checkpoint_callback=callback,
        )
        state = json.loads(json_path.read_text(encoding="utf-8"))
        assert state["completed_bootstraps"] == 1
        assert pd.read_csv(csv_path).to_dict(orient="records") == boot.to_dict(orient="records")
    print("Checkpoint smoke test passed.", flush=True)


def load_checkpoint_records(
    n_boot: int,
    seed: int,
    feature_set_names: list[str],
    metrics: list[str],
) -> list[dict]:
    if not CHECKPOINT_SAMPLES.exists() or not CHECKPOINT_STATE.exists():
        raise SystemExit("No complete stage-61 checkpoint is available to resume.")
    state = json.loads(CHECKPOINT_STATE.read_text(encoding="utf-8"))
    expected = {
        "analysis": "absolute_effect",
        "n_boot": n_boot,
        "seed": seed,
        "feature_set_names": feature_set_names,
        "metrics": metrics,
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise SystemExit("Checkpoint parameters do not match this requested stage-61 run.")
    records = pd.read_csv(CHECKPOINT_SAMPLES).to_dict(orient="records")
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--feature-set",
        action="append",
        choices=sorted(FEATURE_SETS),
        help="Feature set to include. Repeatable. Defaults to critique-primary feature sets.",
    )
    ap.add_argument(
        "--metric",
        action="append",
        choices=ALL_METRICS,
        help="Metric to include. Repeatable. Defaults to score_pref_struct.",
    )
    ap.add_argument("--engine", choices=["optimized", "legacy"], default="optimized")
    ap.add_argument("--parity-check", action="store_true", help="Run deterministic engine parity checks only.")
    ap.add_argument(
        "--checkpoint-smoke-test",
        action="store_true",
        help="Exercise atomic checkpoint writes in a temporary directory only.",
    )
    ap.add_argument(
        "--benchmark",
        type=int,
        metavar="N",
        help="Time both engines for N bootstrap draws without writing analysis outputs.",
    )
    ap.add_argument("--resume", action="store_true", help="Resume a matching atomic checkpoint.")
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
        help="Atomically checkpoint optimized full runs every N bootstraps (default: 500).",
    )
    args = ap.parse_args()

    feature_set_names = args.feature_set or DEFAULT_FEATURE_SET_NAMES
    metrics = args.metric or DEFAULT_METRICS

    if args.checkpoint_smoke_test:
        run_checkpoint_smoke_test()
        return

    base = load_base()

    print(f"Items: {len(base)}", flush=True)
    print(f"Bootstraps: {args.n_boot}")
    print(f"Feature sets: {feature_set_names}")
    print(f"Metrics: {metrics}")

    comp = load_compression(base)
    baselines = load_baselines(base)
    all_scores = pd.concat([comp, baselines], ignore_index=True, sort=False)

    print(f"Compression rows: {len(comp)}")
    print(f"Baseline rows: {len(baselines)}")
    print(f"All score rows: {len(all_scores)}")

    indexed = preindex_bootstrap_inputs(base, all_scores, metrics)

    if args.parity_check:
        run_parity_check(all_scores, indexed, feature_set_names, metrics, args.seed)
        return

    if args.benchmark is not None:
        if args.benchmark <= 0:
            raise SystemExit("--benchmark must be positive.")
        benchmark_draws = generate_bootstrap_draws(indexed.item_ids, args.benchmark, args.seed)
        started = time.perf_counter()
        bootstrap_absolute_effects_legacy(
            all_scores, benchmark_draws, feature_set_names, metrics, progress_label="legacy benchmark"
        )
        legacy_seconds = time.perf_counter() - started
        started = time.perf_counter()
        bootstrap_absolute_effects_optimized(indexed, benchmark_draws, feature_set_names, metrics)
        optimized_seconds = time.perf_counter() - started
        scale = 5000 / args.benchmark
        print(f"Legacy benchmark: {legacy_seconds:.2f}s for {args.benchmark} bootstraps")
        print(f"Optimized benchmark: {optimized_seconds:.2f}s for {args.benchmark} bootstraps")
        print(f"Projected legacy 5000-bootstrap runtime: {legacy_seconds * scale / 60:.1f} minutes")
        print(f"Projected optimized 5000-bootstrap runtime: {optimized_seconds * scale / 60:.1f} minutes")
        return

    print("\nObserved absolute effects...")
    observed = observed_absolute_effects(all_scores, feature_set_names, metrics)

    print("\nBootstrap absolute effects...")
    draws = generate_bootstrap_draws(indexed.item_ids, args.n_boot, args.seed)
    if args.engine == "legacy":
        if args.resume:
            raise SystemExit("--resume is available only for the optimized engine.")
        boot = bootstrap_absolute_effects_legacy(all_scores, draws, feature_set_names, metrics)
    else:
        prior_records = (
            load_checkpoint_records(args.n_boot, args.seed, feature_set_names, metrics)
            if args.resume
            else None
        )
        boot = bootstrap_absolute_effects_optimized(
            indexed,
            draws,
            feature_set_names,
            metrics,
            records=prior_records,
            checkpoint_every=args.checkpoint_every,
            checkpoint_callback=checkpoint_callback_factory(
                args.n_boot, args.seed, feature_set_names, metrics
            ),
        )

    summary = summarize_absolute(observed, boot)

    OUT.mkdir(parents=True, exist_ok=True)

    observed_path = OUT / "bootstrap_absolute_effects_observed.csv"
    boot_path = OUT / "bootstrap_absolute_effects_samples.csv"
    summary_path = OUT / "bootstrap_absolute_effects_summary.csv"
    manifest_path = OUT / "bootstrap_absolute_effects_manifest.json"

    observed.to_csv(observed_path, index=False)
    boot.to_csv(boot_path, index=False)
    summary.to_csv(summary_path, index=False)

    manifest = {
        "analysis": "absolute_effect",
        "n_boot": args.n_boot,
        "seed": args.seed,
        "inputs": {
            "targets_wide": str(DATA / "targets_wide.csv"),
            "scores": str(SCORES),
            "phase_a_results": str(PHASE_A_RESULTS),
        },
        "outputs": {
            "observed": str(observed_path),
            "samples": str(boot_path),
            "summary": str(summary_path),
        },
        "feature_sets": {name: FEATURE_SETS[name] for name in feature_set_names},
        "metrics": metrics,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    write_hashes([observed_path, boot_path, summary_path])
    CHECKPOINT_SAMPLES.unlink(missing_ok=True)
    CHECKPOINT_STATE.unlink(missing_ok=True)

    print("\nAbsolute effect summary:")
    show = summary[
        (summary["metric"] == "score_pref_struct")
        & (summary["feature_set"].isin(["other_human_targets", "other_human_plus_surface", "stacked"]))
    ].sort_values(["feature_set", "method"])
    print(show.to_string(index=False))

    print(f"\nWrote {observed_path}")
    print(f"Wrote {boot_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
