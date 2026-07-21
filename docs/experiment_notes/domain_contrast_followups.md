
## Target robustness: Aesthetic Appeal self-domain

Question:
Does the paired generic-vs-supervised domain contrast survive when the target is Aesthetic_Appeal rather than Surprise?

Command:
python pipeline/62_bootstrap_domain_contrast.py \
  --target Aesthetic_Appeal \
  --pool-target Aesthetic_Appeal \
  --generic-metric v_struct \
  --supervised-metric v_pref_struct \
  --n-boot 100 \
  --seed 123 \
  --tag aesthetic_self_smoke

Interpretation:
TBD after inspecting smoke-test output.

Full result, n_boot=5000:

Generic absolute Aesthetic_Appeal effects remain unresolved:
- Accessible Gutenberg / matched-other: observed rho = -0.072, CI [-0.440, +0.304]
- Accessible Gutenberg / word-shuffle: observed rho = -0.090, CI [-0.430, +0.258]
- Formal Gutenberg / matched-other: observed rho = -0.247, CI [-0.560, +0.091]
- Formal Gutenberg / word-shuffle: observed rho = -0.032, CI [-0.362, +0.293]

Paired supervised-vs-generic domain contrasts all resolve positive:
- Accessible Gutenberg / matched-other: Δrho = +0.547, CI [+0.111, +0.961]
- Accessible Gutenberg / word-shuffle: Δrho = +0.565, CI [+0.165, +0.907]
- Formal Gutenberg / matched-other: Δrho = +0.723, CI [+0.290, +1.081]
- Formal Gutenberg / word-shuffle: Δrho = +0.507, CI [+0.122, +0.849]

Interpretation:
The domain-construction effect is not Surprise-specific. Aesthetic_Appeal shows the same qualitative pattern: generic Gutenberg domains do not yield resolved positive absolute alignment, while human-shaped target-specific domains produce resolved paired improvements over generic domains under all tested control settings.

## Target robustness: Aesthetic_Appeal self-domain, full run

Full result, n_boot=5000:

Generic absolute Aesthetic_Appeal effects remain unresolved:
- Accessible Gutenberg / matched-other: observed rho = -0.072, CI [-0.440, +0.304]
- Accessible Gutenberg / word-shuffle: observed rho = -0.090, CI [-0.430, +0.258]
- Formal Gutenberg / matched-other: observed rho = -0.247, CI [-0.560, +0.091]
- Formal Gutenberg / word-shuffle: observed rho = -0.032, CI [-0.362, +0.293]

Paired supervised-vs-generic domain contrasts all resolve positive:
- Accessible Gutenberg / matched-other: Δrho = +0.547, CI [+0.111, +0.961]
- Accessible Gutenberg / word-shuffle: Δrho = +0.565, CI [+0.165, +0.907]
- Formal Gutenberg / matched-other: Δrho = +0.723, CI [+0.290, +1.081]
- Formal Gutenberg / word-shuffle: Δrho = +0.507, CI [+0.122, +0.849]

Interpretation:
Aesthetic_Appeal shows the strongest target-robustness pattern so far: generic Gutenberg domains do not yield resolved positive absolute alignment, while human-shaped target-specific domains produce resolved paired improvements over generic domains under all tested control settings.

## Target robustness: Creativity self-domain, full run

Full result, n_boot=5000:

Generic absolute Creativity effects:
- Accessible Gutenberg / matched-other: observed rho = -0.184, CI [-0.489, +0.143], unresolved
- Accessible Gutenberg / word-shuffle: observed rho = +0.174, CI [-0.201, +0.515], unresolved
- Formal Gutenberg / matched-other: observed rho = -0.352, CI [-0.582, -0.069], resolved negative
- Formal Gutenberg / word-shuffle: observed rho = +0.174, CI [-0.188, +0.501], unresolved

Paired supervised-vs-generic domain contrasts:
- Accessible Gutenberg / matched-other: Δrho = +0.525, CI [+0.125, +0.886], resolved
- Accessible Gutenberg / word-shuffle: Δrho = +0.167, CI [-0.282, +0.616], unresolved
- Formal Gutenberg / matched-other: Δrho = +0.692, CI [+0.235, +1.064], resolved
- Formal Gutenberg / word-shuffle: Δrho = +0.167, CI [-0.312, +0.646], unresolved

Interpretation:
Creativity supports the matched-control version of the domain-construction effect but not the stronger word-shuffle robustness claim. The emerging cross-target result is therefore: human-shaped domains consistently beat generic domains under matched-control paired item bootstrap across Surprise, Aesthetic_Appeal, and Creativity, while word-shuffle robustness varies by target.

## Cross-target specificity: Surprise ↔ Aesthetic_Appeal

Question:
Does a human-shaped domain need to match the evaluation target, or does it transfer across human appraisal targets?

Full result, n_boot=5000:

### Pool Surprise → Eval Aesthetic_Appeal

Supervised rho = +0.259.

Paired supervised-vs-generic contrasts:
- Accessible Gutenberg / matched-other: Δrho = +0.331, CI [-0.124, +0.760], unresolved
- Accessible Gutenberg / word-shuffle: Δrho = +0.349, CI [-0.041, +0.698], unresolved
- Formal Gutenberg / matched-other: Δrho = +0.506, CI [+0.044, +0.892], resolved
- Formal Gutenberg / word-shuffle: Δrho = +0.291, CI [-0.107, +0.656], unresolved

### Pool Aesthetic_Appeal → Eval Surprise

Supervised rho = +0.377.

Paired supervised-vs-generic contrasts:
- Accessible Gutenberg / matched-other: Δrho = +0.671, CI [+0.248, +1.017], resolved
- Accessible Gutenberg / word-shuffle: Δrho = +0.301, CI [-0.100, +0.703], unresolved
- Formal Gutenberg / matched-other: Δrho = +0.820, CI [+0.465, +1.096], resolved
- Formal Gutenberg / word-shuffle: Δrho = +0.361, CI [-0.029, +0.737], unresolved

Interpretation:
Cross-target transfer exists but is weaker and asymmetric. Aesthetic_Appeal transfers to Surprise better than Surprise transfers to Aesthetic_Appeal. This supports a shared human-appraisal manifold with target-specific sharpening, rather than fully target-specific silos or a completely target-agnostic human-domain effect.

## Consolidated domain-contrast matrix

Artifacts:
- results/tables/domain_contrast_all_controls.csv
- results/tables/domain_contrast_target_pair_matrix.csv
- results/tables/domain_contrast_target_pair_matrix.md
- results/tables/domain_contrast_by_control_family.csv
- results/tables/domain_contrast_by_control_family.md

Main empirical pattern:
1. The matched-control paired domain contrast is the robust result. Across self-domain and cross-target settings, human-shaped domains consistently outperform generic Gutenberg domains under matched-control paired item bootstrap.
2. Word-shuffle controls are stricter and much less stable. They resolve primarily in the cleanest self-domain settings, especially Aesthetic_Appeal.
3. Aesthetic_Appeal behaves like the broadest appraisal target: Aesthetic_Appeal-domain readouts transfer to Surprise and Creativity under matched controls, while Surprise-domain and Creativity-domain readouts transfer less cleanly into Aesthetic_Appeal.
4. Generic absolute effects remain weak, unresolved, or negative. The paper should therefore avoid claiming that generic compression has a resolved positive appraisal signal.
5. The final framing should emphasize domain construction: the human signal enters through the construction of D, not through generic compression alone.

Current best paper claim:
Human-shaped compression domains produce robust paired improvements over generic literary domains under matched-control item-level bootstrap across multiple human appraisal targets. The effect is strongest for target-matched domains and for Aesthetic_Appeal, with weaker but positive cross-target transfer. Word-shuffle robustness is target-dependent and should be reported as a stricter secondary check.

## Observer-family robustness check: Aesthetic_Appeal

Artifact:
- results/analyses/bootstrap_observer_family_summary_aesthetic_observer.csv

Question:
Does the supervised human-shaped Aesthetic_Appeal signal remain positive under GPT-2-family observers?

This is an observer-family absolute-alignment check over existing prefcontrast artifacts. It is not yet the full paired generic-vs-human-domain Stage 62 contrast, because GPT-2 Gutenberg and GPT-2 kfold-surface artifacts are not present.

Results, n_boot=5000:
- gpt2: observed mean rho = +0.235, CI [-0.078, +0.511], resolved = False
- gpt2-medium: observed mean rho = +0.283, CI [-0.045, +0.562], resolved = False

Interpretation: The GPT-2-family observers show positive but unresolved alignment. Observer-family robustness is suggestive but underpowered/unresolved in this narrower absolute-alignment check.

## Same-form domain contrast follow-up

Motivation:
A follow-up critique identified that the current paired domain contrast compares different score forms:
- human-shaped supervised side: `v_pref_struct`, a high-minus-low contrastive score
- generic side: `v_struct`, a single-domain structural score

Therefore the existing paired contrast should be interpreted as human-shaped contrastive construction versus generic single-domain construction, not as a clean isolated test of human-labeled domain construction alone.

Decisive next analysis:
Run a same-form contrast where both sides use `v_pref_struct`:
- human-labeled high/low pools versus nonhuman high/low pools
- same observer
- same metric
- same items
- same bootstrap resamples
- report matched-control and word-shuffle separately

Candidate nonhuman pool axes:
- random high/low splits
- length-based high/low splits
- surface-feature high/low splits
- period/style/canon-derived splits if artifacts exist

Interpretation rule:
Do not strengthen the manuscript's domain-construction mechanism claim unless the same-form contrast resolves, preferably under word-shuffle controls. If the same-form contrast does not resolve, reframe the paper around the stronger boundary result: generic literary compression fails, formal generic compression can anti-align, and human-shaped contrastive construction remains suggestive but not isolated from contrastive form.

## Same-form Surprise vs surface-pool contrast result

Command:
`python pipeline/64_bootstrap_same_form_domain_contrast.py --human-label Surprise --null-label Surprise_surfacepool --metric v_pref_struct --n-boot 5000 --seed 123 --tag surprise_surfacepool`

Purpose:
Test the critique that the previous paired domain contrast mixed score forms:
- human-shaped side used `v_pref_struct`
- generic side used `v_struct`

This follow-up compares two same-form contrastive artifacts:
- human-labeled Surprise `v_pref_struct`
- nonhuman/surface-pool Surprise `v_pref_struct`

Result:
- human Surprise rho = +0.518
- surfacepool Surprise rho = +0.481
- human minus surfacepool diff = +0.037
- 95% bootstrap CI = [-0.179, +0.254]
- n_boot = 5000
- resolved = False

Interpretation:
The same-form contrast does not resolve. This supports the critique that the previous large paired domain-contrast effects should not be interpreted as a clean isolated domain-construction mechanism. Human-labeled contrastive construction aligns positively, but in this available same-form comparison it does not significantly outperform the surface-pool contrastive construction.

Manuscript implication:
Downshift the headline. The strongest resolved result is now the generic-domain failure / anti-alignment result. The human-shaped domain result should be framed as positive but not isolated from contrastive construction at n=36.
