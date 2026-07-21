# Repository instructions

This is a research-paper artifact, not a product repository. Preserve the
negative-with-relocation result:

> Generic and label-free compression-based readouts do not recover human
> creative appraisal. Strong LitBench performance is transductive, same-prompt,
> preference-labeled class probing; the selector remains human.

## Non-negotiable framing

- Do not restore the obsolete Phase A claim that a human-shaped D “works” as a
  label-free or transferable selector.
- Call the language-model result **conditional cross-predictability**.
- Treat directional candidate-to-domain comparison and operator choice as
  load-bearing. MiniLM top-3 exceeds the LM on the identical 1,155 rows.
- Keep the 87.7% result marked as transductive same-prompt
  preference-labeled probing, not an inductive capability comparison.
- Keep the surface-format baseline visible.
- Do not silently substitute the conditional-LM surface row (0.602597) for the
  canonical exact-overlap embedding-summary baseline (0.607792).

## Source and artifact rules

- paper/main.md is the sole manuscript source. Generate paper/main.tex with
  python scripts/build_paper.py --tex; never maintain rival prose.
- Generate Figure 1 only with scripts/make_primary_figure.py. It reads tracked
  result CSVs and asserts the canonical filters and values.
- Do not hand-edit paper/figures/figure1_data.json or the Figure 1 assets.
- Make reproduce is the Phase A-only backward-compatible pipeline. LitBench
  recomputation is available only through explicit reproduce-litbench targets;
  never run concurrent reproductions or overwrite outputs while one is active.

## Publication guardrail

The license is unresolved. Do not invent one, state one, publish, push, stage,
or commit without Eduardo's explicit approval.
