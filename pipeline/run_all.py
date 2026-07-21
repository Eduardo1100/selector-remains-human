from __future__ import annotations

import subprocess
import sys
from pathlib import Path


STAGES = [
    "00_prepare_data.py",
    "10_build_domains.py",
    "20_score_readouts.py",
    "30_compute_controls.py",
    "40_residualize.py",
    "50_permutation.py",
    "60_bootstrap.py",
    ("61_bootstrap_absolute_effects.py", ["--n-boot", "5000", "--seed", "123"]),
    "90_make_tables.py",
    "91_make_figures.py",
    "99_diff_against_scaffold.py",
]


def main() -> None:
    root = Path(__file__).resolve().parent
    for stage in STAGES:
        if isinstance(stage, tuple):
            stage_name, args = stage
        else:
            stage_name, args = stage, []

        path = root / stage_name
        print(f"\n=== Running {stage_name} ===")
        subprocess.run([sys.executable, str(path), *args], check=True)


if __name__ == "__main__":
    main()
