#!/usr/bin/env python3
"""Validate the deliberately minimal CFF metadata and publication prerequisites."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def validate_cff(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {"cff-version", "message", "title", "authors", "license"}
    extra = set(data) - allowed
    assert not extra, f"Unsupported or unverified CFF fields: {sorted(extra)}"
    assert data["cff-version"] == "1.2.0"
    assert isinstance(data["message"], str) and data["message"]
    assert isinstance(data["title"], str) and data["title"]
    assert data["authors"] == [{"family-names": "Cortes", "given-names": "Eduardo"}]
    assert data["license"] == "CC-BY-4.0"


def validate_public_license_notices() -> None:
    required_links = {
        ROOT / "README.md": ["(LICENSE)", "(LICENSE-CONTENT.md)"],
        ROOT / "docs/PUBLICATION.md": ["(../LICENSE)", "(../LICENSE-CONTENT.md)"],
        ROOT / "site/index.html": ['href="LICENSE"', 'href="LICENSE-CONTENT.md"'],
    }
    for path, links in required_links.items():
        text = path.read_text(encoding="utf-8")
        for link in links:
            assert link in text, f"Broken or missing license link {link} in {path.relative_to(ROOT)}"


def validate_no_placeholders(paths: list[Path]) -> None:
    prohibited = ("pending owner-approved", "paper url: pending", "placeholder paper url")
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        found = [marker for marker in prohibited if marker in text]
        assert not found, f"Unresolved publication placeholder in {path.relative_to(ROOT)}: {found}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", action="store_true", help="Also require the generated self-contained site bundle.")
    args = parser.parse_args()
    validate_cff(ROOT / "CITATION.cff")
    validate_public_license_notices()
    validate_no_placeholders([
        ROOT / "README.md", ROOT / "docs/PUBLICATION.md", ROOT / "site/index.html",
        ROOT / "site/publication_README.md",
    ])
    if args.bundle:
        required = [
            ROOT / "site/dist/index.html", ROOT / "site/dist/README.md",
            ROOT / "site/dist/essay.md", ROOT / "site/dist/CITATION.cff",
            ROOT / "site/dist/LICENSE", ROOT / "site/dist/LICENSE-CONTENT.md",
            ROOT / "site/dist/paper/main.md", ROOT / "site/dist/paper/main.pdf",
            ROOT / "site/dist/assets/figure1_data.json", ROOT / "paper/main.pdf",
        ]
        missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
        assert not missing, f"Missing publication bundle files: {missing}"
        validate_no_placeholders([ROOT / "site/dist/index.html", ROOT / "site/dist/README.md"])
    print("Publication metadata checks passed.")


if __name__ == "__main__":
    main()
