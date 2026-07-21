from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCORE_DIR = ROOT / "results" / "scores" / "phase_a_eval_scores"
OUT_DIR = ROOT / "results" / "analyses"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_prefcontrast_name(path: Path) -> dict:
    name = path.name
    stem = name.removesuffix(".csv").removesuffix(".correlations")
    info = {
        "path": str(path),
        "filename": name,
        "is_correlations": name.endswith(".correlations.csv"),
        "observer": None,
        "label": None,
        "control_from_filename": None,
        "seed": None,
        "d_n": None,
        "n_rows": None,
        "columns": None,
        "control_mode_values": None,
        "metrics_available": None,
    }

    m = re.match(
        r"vscore_(?P<observer>.+?)_prefcontrast_chaudhuri_"
        r"(?P<label>.+?)_(?P<control>matchedctrl|wordctrl)_(?P<rest>.+)\.csv$",
        name,
    )
    if not m:
        return info

    info["observer"] = m.group("observer")
    info["label"] = m.group("label")
    info["control_from_filename"] = m.group("control")
    rest = m.group("rest")

    sm = re.search(r"_seed(?P<seed>\d+)_", rest)
    dm = re.search(r"_dn(?P<dn>\d+)", rest)
    if sm:
        info["seed"] = int(sm.group("seed"))
    if dm:
        info["d_n"] = int(dm.group("dn"))

    try:
        df = pd.read_csv(path, nrows=1000)
        info["n_rows"] = len(df)
        info["columns"] = list(df.columns)
        if "control_mode" in df.columns:
            info["control_mode_values"] = sorted(map(str, df["control_mode"].dropna().unique()))
        if "metric" in df.columns:
            info["metrics_available"] = sorted(map(str, df["metric"].dropna().unique()))
        else:
            info["metrics_available"] = [
                c for c in ["v_pref_struct", "v_pref_raw", "v_high_struct", "v_low_struct"]
                if c in df.columns
            ]
    except Exception as exc:
        info["read_error"] = repr(exc)

    return info


def main() -> None:
    files = sorted(SCORE_DIR.glob("vscore_*_prefcontrast_chaudhuri_*.csv"))
    rows = [parse_prefcontrast_name(p) for p in files]
    df = pd.DataFrame(rows)

    inventory_path = OUT_DIR / "same_form_prefcontrast_inventory.csv"
    df.to_csv(inventory_path, index=False)

    usable = df[
        (df["is_correlations"] == False)
        & df["label"].notna()
    ].copy()

    summary = (
        usable.groupby(["observer", "label", "control_from_filename"], dropna=False)
        .agg(
            n_files=("path", "count"),
            seeds=("seed", lambda x: ",".join(map(str, sorted(set(v for v in x if pd.notna(v)))))),
            d_ns=("d_n", lambda x: ",".join(map(str, sorted(set(v for v in x if pd.notna(v)))))),
            control_modes=("control_mode_values", lambda x: json.dumps(sorted(set(sum((v if isinstance(v, list) else [] for v in x), []))))),
        )
        .reset_index()
        .sort_values(["observer", "label", "control_from_filename"])
    )

    summary_path = OUT_DIR / "same_form_prefcontrast_inventory_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {inventory_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
