#!/usr/bin/env python3
"""Verify Manifold brand provenance without modifying publication evidence."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "brand-provenance.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(24)
    assert signature[:8] == b"\x89PNG\r\n\x1a\n", f"Not a PNG: {path}"
    return struct.unpack(">II", signature[16:24])


def main() -> None:
    manifest = json.loads(PROVENANCE.read_text(encoding="utf-8"))
    assert manifest["brand_version"] == "manifold-brand@1.0.0"
    assert manifest["canonical_source"] == "https://manifoldarena.com"

    for asset in manifest["assets"]:
        path = ROOT / asset["local_asset_path"]
        assert path.is_file(), f"Missing brand asset: {path.relative_to(ROOT)}"
        assert sha256(path) == asset["sha256"], f"Brand asset drift: {path.relative_to(ROOT)}"
        assert png_dimensions(path) == (asset["width"], asset["height"]), (
            f"Brand asset dimensions drift: {path.relative_to(ROOT)}"
        )

    for asset in manifest["protected_research_assets"]:
        path = ROOT / asset["path"]
        assert path.is_file(), f"Missing protected research asset: {path.relative_to(ROOT)}"
        assert sha256(path) == asset["sha256"], (
            f"Protected research asset drift: {path.relative_to(ROOT)}"
        )

    page = (ROOT / "site/index.html").read_text(encoding="utf-8")
    required = (
        "Manifold Research",
        "The Selector Remains Human",
        'content="manifold-brand@1.0.0"',
        "https://manifoldarena.com",
        "assets/manifold-mark.png",
        "assets/manifold-favicon.png",
        "assets/manifold-research-social-preview.png",
    )
    for marker in required:
        assert marker in page, f"Missing brand contract marker in site/index.html: {marker}"
    assert "Selector research" not in page
    assert "selector-mark" not in page, "Retired CSS-drawn mark is still present."
    assert "--violet" not in page, "Violet is not a general research-surface decoration."

    preview = ROOT / "site/assets/manifold-research-social-preview.png"
    assert preview.is_file(), "Missing branded social preview."
    assert png_dimensions(preview) == (1200, 630), "Social preview must be 1200x630."
    preview_source = (ROOT / "site/assets/manifold-research-social-preview.svg").read_text(
        encoding="utf-8"
    )
    assert 'href="manifold-mark.png"' in preview_source, (
        "Social preview must use the exact local canonical mark."
    )

    print("Manifold Research brand checks passed.")


if __name__ == "__main__":
    main()
