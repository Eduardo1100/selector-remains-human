# Same-form domain controls plan

## Purpose

The current manuscript branch establishes a negative-boundary result:

1. Generic literary compression does not recover human poetic appraisal.
2. Formal generic compression can anti-align with human ratings.
3. Human-shaped contrastive readouts are positive, but the current generic-vs-human contrast is mixed-form.
4. The available same-form Surprise comparison, human-labeled `v_pref_struct` versus `Surprise_surfacepool` `v_pref_struct`, is positive but unresolved.

The next experiment must isolate whether human labels in the held-out domain add signal beyond contrastive construction itself.

## Decisive comparison

Use the same functional form on both sides:

\[
\rho(v_{\mathrm{pref\_struct}}^{\mathrm{human}}, y)
-
\rho(v_{\mathrm{pref\_struct}}^{\mathrm{nonhuman}}, y)
\]

Both sides must use:

- same observer;
- same item set;
- same score form, ideally `v_pref_struct`;
- same bootstrap resamples;
- same control family where possible.

## Candidate nonhuman contrastive controls

1. Existing `Surprise_surfacepool`
   - already run
   - matched-control only
   - result: positive but unresolved

2. Length-based high/low pools
   - high pool: longest poems in training fold
   - low pool: shortest poems in training fold
   - tests whether surface length contrast explains the apparent human-label signal

3. Predictability-based high/low pools
   - high pool: high LM predictability / low NLL
   - low pool: low LM predictability / high NLL
   - tests whether the signal is ordinary model typicality

4. Random high/low pools
   - several random split seeds
   - tests whether contrastive construction alone creates apparent alignment

5. Surface-feature composite pools
   - split using word length, line count, punctuation, or other frozen surface features if available
   - tests whether shallow surface axes recover the same signal

## Interpretation rule

If human-labeled `v_pref_struct` does not significantly outperform nonhuman `v_pref_struct`, do not claim an isolated human-domain-construction mechanism.

If human-labeled `v_pref_struct` does outperform nonhuman `v_pref_struct`, especially under word-shuffle controls or across a second dataset, then the domain-construction claim can be strengthened.

## Publication decision

For the current 36-poem corpus, this should be treated as an extension experiment. The existing manuscript should remain a negative-boundary paper unless this same-form analysis resolves.

## Same-form Surprise vs surfacepool across observers

We extended the same-form Surprise comparison across all available GPT-2-family observer artifacts.

Results:
- DistilGPT-2: human rho +0.518, surfacepool rho +0.481, diff +0.037, CI [-0.179,+0.254], unresolved.
- GPT-2: human rho +0.537, surfacepool rho +0.476, diff +0.061, CI [-0.159,+0.290], unresolved.
- GPT-2-medium: human rho +0.497, surfacepool rho +0.455, diff +0.042, CI [-0.155,+0.243], unresolved.

Interpretation:
The same-form human-labeled Surprise contrast is consistently positive by point estimate, but it does not significantly outperform the available surface-pool contrast for any observer. This reinforces the negative-boundary manuscript framing. The current evidence does not isolate human labels in D from contrastive construction. Further progress requires generating new same-form nonhuman controls such as randompool, lengthpool, or nllpool, ideally with both matched-control and word-shuffle variants.
