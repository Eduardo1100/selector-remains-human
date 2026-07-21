# Reviewer Critique Action Plan

## Core reframe

The paper should be framed primarily as a negative boundary test: label-free compression over generic literary domains does not recover human poetic appraisal in Phase A.

The supervised preference-shaped-domain result should be framed as a diagnostic contrast, not as the main contribution. Once human labels are used to construct D, the task becomes supervised contrastive probing, and the agreement among compression, TF-IDF, and embedding readouts shows that the label structure is readable by multiple similarity metrics.

## Required manuscript changes

1. Retitle around the negative result.
2. Rewrite the abstract so the generic-D failure is the main contribution.
3. Replace broad phrases like "appraisal structure" with target-specific language such as "held-out Surprise structure" or "residual Surprise."
4. Treat run-level p-values as internal stability diagnostics only.
5. Apply poem-level bootstrap uncertainty to absolute positive claims, not only compression-vs-baseline differences.
6. Separate control regimes and flag item-NLL controls as mechanistically asymmetric to compression.
7. Reframe matched-other as likely variance normalization first, with compression-specific structure unresolved.
8. Downgrade "metric convergence" because the three readouts are all similarity kernels over the same high-minus-low contrast.
9. Clarify that the observer population is single-lineage GPT-2-family, not a diverse compression population.
10. Present Porter and Machery as the highest-value replication target rather than background context.

## Highest-value additional analyses

1. Poem-level bootstrap CI for the absolute stacked compression Surprise effect.
2. Generic-D uncertainty test: distinguish "fails to align" from "anti-aligns."
3. Report convergence excluding item-NLL controls.
4. Show Aesthetic_Appeal / preference-target results directly, even if weaker.
5. If time allows, run Porter and Machery as an actual replication.

## Next implemented analysis

Add absolute poem-level bootstrap intervals for the main positive effects, not only pairwise compression-vs-baseline differences. Required rows:

- compression, Surprise, score_pref_struct, other_human_targets;
- compression, Surprise, score_pref_struct, stacked;
- TF-IDF and embedding analogues for the same control regimes if available.

These intervals determine whether the positive supervised probe claim itself is resolved at item level, rather than only whether compression beats baselines.
