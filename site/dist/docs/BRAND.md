# Manifold Research brand profile

Brand version: `manifold-brand@1.0.0`

Synchronized: `2026-07-27`

Canonical source: [Manifold](https://manifold-official.eduardo11.chatgpt.site)

Canonical manifest:
[`/brand/manifest.json`](https://manifold-official.eduardo11.chatgpt.site/brand/manifest.json)

This repository is a downstream Manifold surface. The official Manifold site is
the source of truth for shared assets, tokens, naming, and behavior.
`brand-provenance.json` records the exact local copies and checksums.

## Surface role

- Publisher: **Manifold Research**
- Paper title: **The Selector Remains Human**
- Profile: editorial research observatory
- Voice: evidence-led, bounded, sober, and explicit about what is not
  established

The paper title is not a publisher name and must remain unchanged. Shared
identity does not make this page a product landing page: editorial serif
headings, evidence labels, scientific restraint, and the original figures are
deliberate surface-specific choices.

## Asset rules

`site/assets/manifold-mark.png` and
`site/assets/manifold-favicon.png` are byte-for-byte copies of the canonical
assets. The mark contains raster-baked geometry and color. Never trace, redraw,
recolor, rotate, stretch, or approximate it in CSS or SVG.

The scientific figures in `paper/figures/` are research evidence, not brand
artwork. Brand framing may surround them, but their bytes, colors, axes, marks,
intervals, and native composition must not change.

## Token and behavior contract

The site uses canonical Manifold inks, text colors, cyan and ember atmosphere,
lines, status colors, luminous-edge actions, radii, focus treatment, and
reduced-motion behavior. Violet is not used as general decoration. The
editorial serif is limited to titles and human-judgment passages.

On narrow screens, publisher identity remains visible and research figures
scroll inside a labeled container instead of shrinking below legibility. The
minimum supported content width is 320 px.

Run `python scripts/check_brand.py` to verify brand provenance, asset bytes,
dimensions, metadata, and the scientific-figure immutability guard.
