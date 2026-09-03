#!/usr/bin/env python3
"""Build and verify a self-contained, static publication bundle."""

from __future__ import annotations

import argparse
import shutil
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DIST = SITE / "dist"
ASSETS = [
    "paper/figures/selector_remains_human_figure1a.svg",
    "paper/figures/selector_remains_human_figure1a.png",
    "paper/figures/selector_remains_human_figure1b.svg",
    "paper/figures/selector_remains_human_figure1b.png",
    "paper/figures/selector_remains_human_social_preview.png",
    "paper/figures/figure1_data.json",
]
BRAND_ASSETS = [
    "site/assets/manifold-mark.png",
    "site/assets/manifold-favicon.png",
    "site/assets/manifold-research-social-preview.svg",
    "site/assets/manifold-research-social-preview.png",
]
DOCUMENTS = [
    "LICENSE",
    "LICENSE-CONTENT.md",
    "paper/main.md",
    "paper/main.pdf",
    "essay.md",
    "CITATION.cff",
    "docs/reproducibility.md",
    "docs/PUBLICATION.md",
    "docs/BRAND.md",
    "site/publication_README.md",
    "brand-provenance.json",
]


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.links.append(value)


def copy_file(relative: str, destination: str | None = None) -> None:
    source = ROOT / relative
    if not source.exists():
        raise SystemExit(f"Missing publication input: {source}")
    target = DIST / (destination or relative)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def check() -> None:
    required = [
        DIST / "index.html", DIST / "essay.md", DIST / "CITATION.cff", DIST / "README.md",
        DIST / "LICENSE", DIST / "LICENSE-CONTENT.md", DIST / "paper/main.md", DIST / "paper/main.pdf",
        DIST / "assets/selector_remains_human_figure1a.svg",
        DIST / "assets/selector_remains_human_figure1a.png",
        DIST / "assets/selector_remains_human_figure1b.svg",
        DIST / "assets/selector_remains_human_figure1b.png",
        DIST / "assets/selector_remains_human_social_preview.png", DIST / "assets/figure1_data.json",
        DIST / "assets/manifold-mark.png", DIST / "assets/manifold-favicon.png",
        DIST / "assets/manifold-research-social-preview.svg",
        DIST / "assets/manifold-research-social-preview.png",
        DIST / "site.webmanifest", DIST / "brand-provenance.json", DIST / "docs/BRAND.md",
    ]
    missing = [str(path.relative_to(DIST)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Incomplete site bundle: {', '.join(missing)}")
    parser = LinkCollector()
    parser.feed((DIST / "index.html").read_text(encoding="utf-8"))
    broken = []
    for link in parser.links:
        if "://" in link or link.startswith("#"):
            continue
        if link.startswith("../"):
            broken.append(f"parent path not allowed: {link}")
            continue
        target = DIST / link.split("#", 1)[0]
        if not target.exists():
            broken.append(link)
    if broken:
        raise SystemExit(f"Broken site links: {', '.join(broken)}")
    landing_page = (DIST / "index.html").read_text(encoding="utf-8")
    if "placeholder" in landing_page.lower() or "pending owner-approved" in landing_page.lower():
        raise SystemExit("Unresolved paper URL placeholder in site bundle.")
    print(f"Self-contained site bundle verified: {DIST.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Validate an existing site/dist bundle.")
    args = parser.parse_args()
    if args.check:
        check()
        return
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir(parents=True)
    copy_file("site/index.html", "index.html")
    for relative in ASSETS:
        copy_file(relative, "assets/" + Path(relative).name)
    for relative in BRAND_ASSETS:
        copy_file(relative, "assets/" + Path(relative).name)
    for relative in DOCUMENTS:
        destination = "README.md" if relative == "site/publication_README.md" else relative
        copy_file(relative, destination)
    copy_file("site/site.webmanifest", "site.webmanifest")
    check()


if __name__ == "__main__":
    main()
