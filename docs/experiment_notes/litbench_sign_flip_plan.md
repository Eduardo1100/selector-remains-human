# LitBench sign-flip experiment plan

## Motivation

The poetry results now support a negative-boundary interpretation:

- generic literary compression does not recover human poetic appraisal as a resolved positive signal;
- formal generic compression can anti-align with ratings;
- human-shaped contrastive construction is positive but not isolated from contrastive form;
- same-form human Surprise vs Surprise_surfacepool is positive but unresolved across DistilGPT-2, GPT-2, and GPT-2-medium.

The critic's key point is that the high surfacepool score is itself mechanistically informative. A nonhuman surface-feature contrast recovered almost as much held-out Surprise structure as the human-labeled contrast. This suggests that the positive signal may be mediated by surface-correlated contrastive structure rather than uniquely human aesthetic semantics.

LitBench lets us test a broader and more falsifiable hypothesis:

> Compression/predictability measures conventionality, fluency, accessibility, or typicality. Whether that signal aligns or anti-aligns with human preference depends on the evaluation population and task.

## Core sign-flip hypothesis

In the current poetry-aesthetic setting:

- contemporary raters may penalize conventionality;
- generic compression therefore fails or anti-aligns;
- higher-rated poems can be more predictable, showing that human appraisal is not simple LM surprise.

In LitBench:

- labels are pairwise creative-writing preferences derived from Reddit upvotes;
- crowd preference may reward accessibility, fluency, readability, and conventionality more than expert-ish poetry appraisal does;
- therefore generic compression/predictability may flip sign and positively align with preference.

## Dataset

Primary dataset:

- SAA-Lab/LitBench-Test
- SAA-Lab/LitBench-Train

LitBench provides a paired creative-writing preference benchmark. The paper reports:

- 43,827-pair training corpus;
- held-out debiased human-labeled test set;
- Claude-3.7-Sonnet as strongest off-the-shelf judge at 73%;
- trained Bradley-Terry and generative reward models at 78%.

## First-stage tests

### Test 1: Dataset ingest

Load LitBench test and train splits.

Record:

- column schema;
- number of pairs;
- available IDs;
- whether full story text is included or whether only comment IDs are included;
- average story length if text is available.

### Test 2: Pairwise generic predictability baseline

For each pair:

- compute LM negative log-likelihood or average token NLL for chosen and rejected texts;
- convert to a pairwise prediction:
  - prefer lower NLL / more predictable text;
  - optionally prefer higher NLL / more surprising text;
- report pairwise agreement with LitBench labels.

This directly tests whether crowd preference rewards predictability/accessibility.

### Test 3: Length and surface baselines

For each pair:

- prefer longer text;
- prefer shorter text;
- prefer lower type-token ratio;
- prefer higher type-token ratio;
- prefer simpler readability / surface fluency if cheaply available.

This tests whether any compression signal is just length or surface structure.

### Test 4: Contrastive domain construction

Generate same-form nonhuman controls:

- randompool v_pref_struct;
- lengthpool v_pref_struct;
- nllpool v_pref_struct;
- surface-feature-pool v_pref_struct.

Then compare against any human/preference-shaped domain construction:

- same score form;
- same observer;
- same pairwise evaluation;
- bootstrap over pairs or prompts.

## Evaluation metrics

Primary:

- pairwise accuracy against human preference labels.

Secondary:

- bootstrap confidence interval over pairwise accuracy;
- prompt-clustered bootstrap if prompt IDs are available;
- difference from 50%;
- difference from length baseline;
- difference from published judge/reward baselines when comparable.

## Interpretation rule

If generic predictability/compression positively predicts LitBench crowd preference while anti-aligning with poetry-aesthetic ratings, this supports the sign-flip theory:

> compression tracks conventionality / fluency / typicality, and the evaluative setting determines whether that signal is rewarded or penalized.

If generic predictability fails on both poetry and LitBench, then the generic compression result is a narrower negative finding.

If contrastive human/preference-shaped domains beat same-form nonhuman controls, then domain construction becomes stronger.

If they do not, the mechanism remains surface/contrastive rather than human-semantic.

## Guardrails

Do not claim this is a direct poetry replication. LitBench is a generalization/stress-test dataset:

- construct: creative-writing preference, not lab-rated poetic appraisal;
- labels: Reddit-derived preference, not expert aesthetic ratings;
- genre: stories, not poems;
- advantage: scale, pairwise labels, published baselines, longer texts.

## Immediate success criterion

The first checkpoint is not to beat LitBench reward models. The first checkpoint is to determine whether simple predictability/compression has a stable and interpretable directional relationship with LitBench preference.


## First held-out LitBench baselines

We scored the complete held-out LitBench test artifact:

- dataset: SAA-Lab/LitBench-Test-IDs-Complete
- n = 2,480 chosen/rejected pairs
- full prompt, chosen_story, rejected_story, metadata, comment IDs, and post IDs are available.

### Surface baselines

The strongest non-upvote surface baselines were:

- prefer_more_paragraphs: 58.87%, CI [56.93%, 60.77%]
- prefer_more_newlines: 58.02%, CI [56.13%, 59.92%]
- prefer_more_punct: 55.60%, CI [53.67%, 57.50%]

Raw length was near chance:

- prefer_more_words: 50.93%, CI [49.03%, 52.82%]
- prefer_more_chars: 50.93%, CI [48.99%, 52.78%]

Interpretation:
LitBench preference is not simply a preference for longer stories. It is more strongly associated with formatting and structural organization: paragraphs, line breaks, and punctuation.

### DistilGPT-2 average NLL baseline

We scored each chosen/rejected story with DistilGPT-2 using sliding-window average token negative log-likelihood.

Results:

- prefer_lower_avg_nll: 54.27%, CI [52.34%, 56.21%]
- prefer_higher_avg_nll: 45.73%, CI [43.79%, 47.66%]
- prefer_higher_total_nll: 51.33%, CI [49.35%, 53.23%]
- prefer_lower_total_nll: 48.67%, CI [46.77%, 50.65%]

Interpretation:
Preferred LitBench stories are more predictable / fluent under DistilGPT-2 by average token NLL. This signal is modest but resolved, and it is stronger than raw length. The total-NLL direction is not clean because total NLL is heavily entangled with length.

This supports the emerging cross-domain mechanism:

> Compression or predictability does not measure literary value directly. It measures fluency, accessibility, conventionality, or typicality. Whether that signal helps or hurts depends on the evaluative population and task.

For Reddit creative-writing preference, predictability/fluency appears mildly rewarded. In the poetry setting, generic formal compression can anti-align with contemporary appraisal, suggesting that raters may penalize conventionality in that domain.

## Combined surface + NLL model

We trained simple cross-validated logistic pairwise models on the held-out complete LitBench test set using pairwise feature differences.

Results:

- surface_format: 60.60%, CI [58.67%, 62.46%]
- nll_avg_only: 55.12%, CI [53.15%, 57.06%]
- surface_plus_avg_nll: 61.29%, CI [59.31%, 63.23%]
- surface_plus_avg_and_total_nll: 61.37%, CI [59.44%, 63.27%]

Paired bootstrap deltas:

- surface_plus_avg_nll - surface_format: +0.69 points, CI [-0.65,+2.02], unresolved.
- surface_plus_avg_and_total_nll - surface_format: +0.77 points, CI [-0.52,+2.10], unresolved.
- surface_format - nll_avg_only: +5.48 points, CI [+3.06,+7.94], resolved.

Interpretation:
Average NLL predicts held-out LitBench preference above chance, but its incremental value beyond formatting/surface organization is not resolved. Formatting and surface organization are the stronger predictors. The sign-flip hypothesis should therefore be softened: LitBench preference rewards readable organization and modestly rewards LM fluency/predictability, rather than being primarily explained by predictability.

## Interpretation correction after critic review

The first LitBench results should be interpreted more conservatively.

The held-out test result confirms a sign-flip for raw average predictability:

- preferred stories have lower DistilGPT-2 average NLL than rejected stories;
- prefer_lower_avg_nll reaches 54.27%, CI [52.34%, 56.21%].

However, this is not yet a test of the paper's central compression-progress quantity V. It is a test of unconditional fluency / typicality.

The combined model also shows that NLL does not add a resolved increment beyond surface formatting:

- surface_format: 60.60%, CI [58.67%, 62.46%]
- surface_plus_avg_nll: 61.29%, CI [59.31%, 63.23%]
- delta: +0.69 points, CI [-0.65,+2.02], unresolved.

Therefore, the honest current LitBench claim is:

> LitBench preference is primarily captured by readable formatting / surface organization among the simple features tested. Raw average predictability is above chance, but its incremental value beyond formatting is unresolved.

The framework-relevant test remains to be run:

> Does compression-progress V predict LitBench preference, especially after residualizing or controlling for formatting?

Next priority:
1. inventory repeated prompts / post IDs / possible domain construction;
2. implement global compression-progress V;
3. implement prompt-conditioned V if the data support it;
4. evaluate V raw and net of formatting;
5. repeat with GPT-2-family observers if feasible.

## Prompt-conditioned V result

We implemented the framework-relevant quantity:

> V(candidate) = prompt-only domain NLL - candidate-conditioned domain NLL

For each LitBench test pair, the domain D was constructed from other chosen stories for the same prompt, excluding the current pair. This creates a leave-pair-out same-prompt domain.

Configuration:

- dataset: SAA-Lab/LitBench-Test-IDs-Complete
- observer: DistilGPT-2
- domain_mode: other_chosen
- min_domain: 2
- max_domain: 3
- max_context_tokens: 512
- max_target_tokens: 384
- eligible rows: 1,385 / 2,480

Direct V sign-rule result:

- predict chosen if V(chosen) > V(rejected)
- accuracy: 63.75%
- CI [61.23%, 66.35%]

Mean V delta:

- mean_v_delta_gain: +0.203
- CI [+0.179,+0.228]

This is substantially stronger than the raw average NLL baseline.

## Prompt-conditioned V net of formatting

We then controlled V against surface/formatting features using cross-validated logistic pairwise models on the same eligible subset.

Results:

- surface_format: 60.00%, CI [57.33%, 62.60%]
- v_only_sign_rule: 63.75%, CI [61.23%, 66.35%]
- v_only_logistic: 63.90%, CI [61.30%, 66.50%]
- surface_plus_v: 69.75%, CI [67.36%, 72.13%]

Paired deltas:

- surface_plus_v - surface_format: +9.75 points, CI [+7.51,+11.99], resolved.
- surface_plus_v - v_only_logistic: +5.85 points, CI [+2.60,+9.10], resolved.
- v_only_sign_rule - surface_format: +3.75 points, CI [-0.07,+7.73], near-resolved.
- v_only_logistic - surface_format: +3.90 points, equivalent paired difference resolved in the logistic comparison.

Interpretation:

Unlike raw average NLL, prompt-conditioned V survives the formatting confound. The strongest current LitBench claim is now:

> On the same-prompt eligible subset, preferred stories are better compression-progress exemplars for other preferred stories under the same prompt, and this signal adds a large resolved increment over formatting features.

Caution:

This result is currently limited to the repeated-prompt eligible subset. It should not yet be generalized to all 2,480 LitBench pairs. The next control is to compare other_chosen domains against other_rejected and all_other domains to test whether V specifically tracks the preferred-response domain or merely prompt-consistency / generic response quality.

## Preferred-minus-rejected domain contrast

We combined the other_chosen and other_rejected prompt-conditioned V runs into a direct domain contrast.

Definitions:

- V_chosen_domain(candidate): compression-progress of candidate for other chosen stories under the same prompt.
- V_rejected_domain(candidate): compression-progress of candidate for other rejected stories under the same prompt.
- domain_specificity(candidate) = V_chosen_domain(candidate) - V_rejected_domain(candidate).
- domain_contrast_delta =
    domain_specificity(chosen_story) - domain_specificity(rejected_story).

The direct sign rule predicts the chosen story when:

> domain_specificity(chosen_story) > domain_specificity(rejected_story)

Configuration:

- dataset: SAA-Lab/LitBench-Test-IDs-Complete
- observer: DistilGPT-2
- domain construction: same-prompt, leave-pair-out
- chosen-domain source: other chosen stories for same prompt
- rejected-domain source: other rejected stories for same prompt
- eligible rows with both domains: 1,155

Results:

- domain_contrast_sign_rule: 75.58%, CI [73.16%, 78.01%]
- domain_specificity_logistic: 76.19%, CI [73.77%, 78.70%]
- surface_format: 60.26%, CI [57.32%, 63.03%]

Paired deltas over formatting:

- domain_contrast_sign_rule - surface_format:
  +15.32 points, CI [+11.52,+19.13], resolved.
- domain_specificity_logistic - surface_format:
  +15.93 points, CI [+12.21,+19.65], resolved.

Continuous effects:

- domain_contrast_delta:
  mean +0.420, CI [+0.377,+0.464]
- delta_chosen_domain:
  mean +0.203, CI [+0.177,+0.231]
- delta_rejected_domain:
  mean -0.216, CI [-0.248,-0.185]

Interpretation:

This is the strongest LitBench result so far. Prompt-conditioned compression-progress separates preferred and rejected response domains within the same prompt. Chosen stories move toward the chosen-response domain; rejected stories move toward the rejected-response domain. The preferred-minus-rejected domain contrast predicts preference at about 76% accuracy and beats formatting by about 15 points.

Caution:

This result is currently limited to the repeated-prompt subset where both chosen and rejected same-prompt domains are available. It is also a label-shaped domain contrast rather than a label-free selector. The next robustness test is observer-family replication with GPT-2 and GPT-2-medium, budget permitting.

## Random same-prompt pool split control

After the critic raised the circularity concern, we ran a same-form nonhuman pool-split control.

Instead of defining domains using preference labels:

- D_preferred = other chosen stories
- D_rejected = other rejected stories

we randomly split same-prompt stories into arbitrary pools A/B and recomputed the same domain-contrast machinery.

Configuration:

- dataset: SAA-Lab/LitBench-Test-IDs-Complete
- observer: DistilGPT-2
- same-prompt random A/B pools
- min_domain: 2
- max_domain: 3
- max_context_tokens: 512
- max_target_tokens: 384
- seed: 123
- eligible rows: 1,161
- bootstrap: 5,000 pair resamples

Random split results:

- random_domain_contrast_sign_rule:
  48.92%, CI [46.08%, 51.85%]
- random_domain_specificity_logistic:
  48.49%, CI [45.56%, 51.25%]
- surface_format:
  59.52%, CI [56.67%, 62.36%]
- surface_plus_random_domain_specificity:
  59.52%, CI [56.68%, 62.36%]

Paired deltas:

- random_domain_contrast_sign_rule - surface_format:
  -10.59 points, CI [-14.56,-6.55]
- random_domain_specificity_logistic - surface_format:
  -11.02 points, CI [-14.99,-7.06]
- surface_plus_random_domain_specificity - surface_format:
  +0.00 points, CI [-0.95,+0.95]

Continuous effects:

- domain_contrast_delta:
  mean +0.0053, CI [-0.0418,+0.0515]
- delta_a:
  mean -0.0057, CI [-0.0351,+0.0227]
- delta_b:
  mean -0.0110, CI [-0.0410,+0.0190]
- chosen_domain_specificity:
  mean +0.0083, CI [-0.0213,+0.0385]
- rejected_domain_specificity:
  mean +0.0030, CI [-0.0299,+0.0368]

Interpretation:

The random same-prompt pool split does not reproduce the preferred-minus-rejected domain contrast. Both the direct random-domain sign rule and the logistic random-domain specificity model are near chance, and random-domain features add nothing to formatting. This suggests that the earlier ~76% preferred/rejected contrast is not merely arbitrary same-prompt pool geometry.

Updated cautious claim:

> Preference-labeled domain construction carries signal beyond generic within-prompt similarity geometry.

Remaining caveat:

This does not yet establish compression-specificity. The next decisive test is to run TF-IDF and embedding kernels on the identical preferred/rejected pools. If those kernels match compression V, the honest claim becomes domain construction rather than compression-specificity.

## TF-IDF kernel control on identical preferred/rejected pools

After the random pool-split control, we tested whether a non-compression similarity kernel could recover the same preferred-minus-rejected domain contrast.

We used the identical same-prompt preferred/rejected pools as the compression V run:

- D_preferred = other chosen stories for the same prompt
- D_rejected = other rejected stories for the same prompt
- min_domain = 2
- max_domain = 3
- eligible rows = 1,155

But instead of candidate-conditioned NLL reduction, we computed TF-IDF cosine similarity to preferred and rejected pool centroids.

Definitions:

- tfidf_domain_specificity(candidate) =
    cosine(candidate, D_preferred_centroid)
    -
    cosine(candidate, D_rejected_centroid)

- tfidf_domain_contrast_delta =
    tfidf_domain_specificity(chosen_story)
    -
    tfidf_domain_specificity(rejected_story)

Results:

- TF-IDF domain_specificity_logistic:
  51.08%, CI [48.22%, 53.85%]
- TF-IDF surface_plus_domain_specificity:
  60.69%, CI [57.75%, 63.55%]
- surface_format:
  60.26%, CI [57.32%, 63.03%]
- TF-IDF domain_contrast_sign_rule:
  11.95%, CI [10.13%, 13.85%]

Paired deltas:

- TF-IDF domain_specificity_logistic - surface_format:
  -9.18 points, CI [-13.16,-5.19]
- TF-IDF surface_plus_domain_specificity - surface_format:
  +0.43 points, CI [-0.52,+1.39], unresolved.

Continuous effects:

- TF-IDF domain_contrast_delta:
  mean +0.000122, CI [-0.000078,+0.000334]
- chosen_domain_specificity:
  mean +0.000460, CI [+0.000123,+0.000835]
- rejected_domain_specificity:
  mean +0.000338, CI [+0.000037,+0.000674]

Interpretation:

TF-IDF does not reproduce the compression V domain-contrast result. The TF-IDF logistic domain-specificity control is near chance, and adding TF-IDF domain specificity to formatting adds no resolved improvement. This weakens the "any kernel recovers the same contrast" objection, at least for lexical TF-IDF.

Updated cautious claim:

> The preferred/rejected domain contrast is not reproduced by arbitrary random pools or by a TF-IDF centroid kernel. Compression V remains substantially stronger than these controls on the same repeated-prompt subset.

Remaining caveat:

This still does not rule out semantic embedding kernels. The next kernel control should use sentence embeddings on the identical pools.

## Embedding kernel control on identical preferred/rejected pools

We next tested whether a semantic embedding kernel could recover the same preferred-minus-rejected domain contrast.

We used the identical same-prompt preferred/rejected pools as the compression V and TF-IDF runs:

- D_preferred = other chosen stories for the same prompt
- D_rejected = other rejected stories for the same prompt
- min_domain = 2
- max_domain = 3
- eligible rows = 1,155

Kernel:

- sentence-transformers/all-MiniLM-L6-v2
- normalized story embeddings
- centroid cosine similarity to D_preferred and D_rejected

Definitions:

- embedding_domain_specificity(candidate) =
    cosine(candidate, D_preferred_centroid)
    -
    cosine(candidate, D_rejected_centroid)

- embedding_domain_contrast_delta =
    embedding_domain_specificity(chosen_story)
    -
    embedding_domain_specificity(rejected_story)

Results:

- embedding domain_specificity_logistic:
  48.92%, CI [46.15%, 51.77%]
- embedding surface_plus_domain_specificity:
  60.00%, CI [57.14%, 62.77%]
- surface_format:
  60.26%, CI [57.32%, 63.03%]
- embedding domain_contrast_sign_rule:
  12.03%, CI [10.13%, 14.03%]

Paired deltas:

- embedding domain_specificity_logistic - surface_format:
  -11.34 points, CI [-15.32,-7.36]
- embedding surface_plus_domain_specificity - surface_format:
  -0.26 points, CI [-1.04,+0.52], unresolved.

Continuous effects:

- embedding domain_contrast_delta:
  mean -0.000040, CI [-0.001302,+0.001220]
- chosen_domain_specificity:
  mean +0.003804, CI [+0.001371,+0.006246]
- rejected_domain_specificity:
  mean +0.003845, CI [+0.001469,+0.006217]

Interpretation:

The MiniLM embedding centroid kernel does not reproduce the compression V domain-contrast result. Like TF-IDF, embedding domain specificity is near chance and adds no improvement over formatting. This weakens the "any similarity kernel recovers the contrast" objection for both lexical and sentence-embedding centroid kernels.

Updated cautious claim:

> On the repeated-prompt LitBench subset, preferred/rejected compression-progress V predicts preference strongly, while random same-prompt pool splits, TF-IDF centroid similarity, and MiniLM embedding centroid similarity do not reproduce the effect.

Remaining caveats:

This still uses label-shaped domains, small same-prompt pools, and a repeated-prompt subset. The next methodological checks are max_domain sweeps and subset-matched baselines.

## Max-domain sweep

We tested whether the preferred/rejected compression-progress V contrast depends on very small same-prompt pools.

Prior configuration:

- min_domain = 2
- max_domain = 3
- eligible overlap = 1,155 pairs
- domain_contrast_sign_rule:
  75.58%, CI [73.16%, 78.01%]
- surface_format:
  60.26%, CI [57.32%, 63.03%]

We then increased the domain cap.

### max_domain = 6

Chosen-domain run:

- other_chosen sign rule:
  65.70%, CI [63.25%, 68.16%]
- mean_v_delta_gain:
  +0.1915, CI [+0.1692,+0.2146]

Rejected-domain run:

- other_rejected direct sign rule:
  30.73%, CI [28.26%, 33.28%]
- mean_v_delta_gain:
  -0.2353, CI [-0.2633,-0.2069]

Preferred-minus-rejected domain contrast:

- domain_contrast_sign_rule:
  79.83%, CI [77.49%, 82.08%]
- domain_specificity_logistic:
  79.31%, CI [76.97%, 81.65%]
- surface_format:
  60.26%, CI [57.32%, 63.03%]

Paired deltas:

- domain_contrast_sign_rule - surface_format:
  +19.57 points, CI [+15.93,+23.29]
- domain_specificity_logistic - surface_format:
  +19.05 points, CI [+15.41,+22.77]

Continuous effects:

- domain_contrast_delta:
  +0.4288, CI [+0.3892,+0.4697]
- delta_chosen_domain:
  +0.1896, CI [+0.1652,+0.2136]
- delta_rejected_domain:
  -0.2393, CI [-0.2694,-0.2106]

### max_domain = 10

Chosen-domain run:

- other_chosen sign rule:
  65.92%, CI [63.47%, 68.45%]
- mean_v_delta_gain:
  +0.1908, CI [+0.1686,+0.2137]

Rejected-domain run:

- other_rejected direct sign rule:
  30.42%, CI [27.95%, 32.97%]
- mean_v_delta_gain:
  -0.2360, CI [-0.2642,-0.2078]

Preferred-minus-rejected domain contrast:

- domain_contrast_sign_rule:
  79.74%, CI [77.40%, 82.08%]
- domain_specificity_logistic:
  79.57%, CI [77.23%, 81.90%]
- surface_format:
  60.26%, CI [57.32%, 63.03%]

Paired deltas:

- domain_contrast_sign_rule - surface_format:
  +19.48 points, CI [+15.84,+23.12]
- domain_specificity_logistic - surface_format:
  +19.31 points, CI [+15.76,+23.03]

Continuous effects:

- domain_contrast_delta:
  +0.4287, CI [+0.3892,+0.4692]
- delta_chosen_domain:
  +0.1887, CI [+0.1646,+0.2125]
- delta_rejected_domain:
  -0.2400, CI [-0.2704,-0.2111]

Interpretation:

The preferred/rejected compression-progress V contrast does not depend on the smallest 2–3 story pools. Increasing max_domain from 3 to 6 strengthens the effect, and max_domain = 10 remains essentially stable. This weakens the small-domain/idiosyncratic-pool objection.

Updated cautious claim:

> On the repeated-prompt LitBench subset, preferred/rejected compression-progress V predicts preference strongly. The effect survives random same-prompt pool controls, is not reproduced by TF-IDF or MiniLM centroid kernels, and remains stable as the same-prompt domain cap increases from 3 to 6 to 10 stories.

Remaining caveats:

The result still uses label-shaped domains and a repeated-prompt subset. The next useful check is subset-matched baselines, especially surface/readability and any available LitBench reward-model baseline on the exact 1,155-pair overlap.

## Critic response after random-split, kernel, and max-domain controls

The critic agreed that the random split control is a real win: arbitrary same-prompt pool geometry collapsed to chance, so the preferred/rejected V result is not merely any within-prompt split. The critic also agreed that the max_domain sweep addresses the tiny-pool objection: the effect strengthens from max_domain=3 to 6 and remains stable at 10.

However, the critic argued that the kernel controls should not be interpreted as proving compression-specificity. The reason is that the TF-IDF and MiniLM controls changed both representation and comparison operator:

- Compression V uses directional, asymmetric, conditional next-token prediction over concatenated domains.
- TF-IDF and MiniLM controls used symmetric centroid cosine similarity.

Therefore, the clean conclusion is not "compression is uniquely special." The cleaner conclusion is that conditional comparison preserves chosen/rejected class structure that centroid cosine does not.

The critic's recommended framing:

> Upvoted and downvoted stories for the same prompt occupy detectably different regions of next-token-predictable structure, and that difference is real, non-arbitrary, and directional.

The critic emphasized that this is still supervised class-separation because D_preferred and D_rejected are constructed from preference labels. The random-split null rules out arbitrary pool geometry, but not label-defined supervised probing.

Next priority:

1. Run subset-matched baselines on the exact 1,155-pair overlap, especially the released LitBench reward model if feasible.
2. Then fix kernel controls with a crossed design:
   - embeddings with non-centroid comparison, such as per-story max or top-k similarity;
   - possibly LM-style centroid/pooled comparison if feasible.
3. Later: label-leakage-free domains, rationale mining, and observer-family replication.

Updated cautious claim:

> On the repeated-prompt LitBench subset, preference-labeled chosen/rejected classes are strongly separable by a directional conditional-prediction probe. The effect is not reproduced by random same-prompt pool splits and is stable across domain caps, but it remains a supervised label-shaped probe rather than label-free value discovery.

## Attempted subset-matched LitBench reward-model baseline

We attempted to run the released LitBench verifier on the exact 1,155-pair domain-contrast overlap.

Configuration:

- reward model: SAA-Lab/Llama8B-CreativeWritingVerifier
- subset: rows from test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10
- n = 1,155 pairs
- device: CPU
- max_length: 2048
- batch_size: 1

Outcome:

The model downloaded successfully, but the local CPU process was killed while loading/running the 15GB model. No reward scores were written.

Interpretation:

The released reward model is likely not feasible on this local CPU setup. We should either run it on a GPU/HPC machine or report this as an attempted but infeasible subset-matched baseline and use local subset-matched baselines in the meantime.

## Critic response after subset-matched local baseline table

The critic argued that the new subset-matched local baseline table did not add independent scientific evidence; it mostly re-tabulated the prior random-pool, centroid-kernel, surface, and compression-V results on the max_domain=10 overlap. The one genuinely new event was negative: the released 8B LitBench verifier failed locally on CPU.

The critic agreed with the cautious framing:

> The current result is a supervised, label-shaped directional class-separation probe, not label-free value discovery and not evidence that compression uniquely predicts human value.

Main corrections:

1. The reward-model baseline is not the immediate scientific bottleneck.
   It answers whether the current score is competitive, but not what the score means.

2. The crossed-kernel control is more urgent.
   Existing TF-IDF and MiniLM controls confound representation with comparison operator:
   - compression V uses directional conditional next-token prediction;
   - TF-IDF/MiniLM controls use symmetric centroid cosine.
   Therefore, the centroid-kernel nulls do not yet establish compression-specificity.

3. The label-leakage caveat remains central.
   D_preferred and D_rejected are constructed from the same preference labels being predicted.
   Random-split controls rule out arbitrary pool geometry, but they do not make the probe label-free.

4. The more neutral name for the current metric is conditional cross-predictability:
   "Does this candidate make pool stories more predictable?"
   Calling it compression-progress V imports extra theoretical loading.

Updated internal framing:

> Upvoted and downvoted same-prompt stories occupy detectably different regions of conditional-prediction space. This structure is directional, non-arbitrary, stable across domain sizes, and not recovered by centroid controls. However, the probe remains supervised because the domains are label-defined.

Next priority:

1. Crossed-kernel operator control:
   embeddings with per-story max similarity, mean top-k similarity, and kNN vote using the identical preference-labeled pools.

2. Label-leakage-reduced domains:
   construct domains from a signal not identical to the held-out test labels being predicted.

3. Hosted/GPU reward-model scoring:
   useful later for calibration, but not the next scientific bottleneck.

4. Rationale mining:
   useful after the core phenomenon is identified.

## Next experiment: crossed embedding operator control

The critic argued that the TF-IDF and MiniLM kernel controls are not clean compression-specificity tests because they changed both representation and comparison operator:

- conditional cross-predictability uses directional/asymmetric candidate-to-pool prediction;
- TF-IDF/MiniLM controls used symmetric centroid cosine.

To test the operator confound, we next run MiniLM embedding controls with non-centroid, per-story comparison operators on the identical preference-labeled pools:

- mean per-story cosine
- max per-story cosine
- top-2 mean cosine
- top-3 mean cosine
- top-5 mean cosine
- kNN vote with k=3
- kNN vote with k=5

This tests whether the prior embedding null was due to centroid pooling rather than the embedding representation itself.

Decision rule:

- If per-story/top-k/kNN embedding operators recover most of the 79–80% compression result, then compression-specificity weakens and the durable finding is directional operator/class-separation.
- If these operators remain near surface/chance, conditional LM cross-predictability becomes more credible as special relative to tested embedding operators.

## Crossed embedding operator control: first full run

We tested the critic's operator-confound objection by replacing centroid cosine with per-story MiniLM embedding operators on the same preference-labeled domain construction.

Operators:

- mean per-story cosine
- max per-story cosine
- top-2 mean cosine
- top-3 mean cosine
- top-5 mean cosine
- kNN vote, k=3
- kNN vote, k=5

Configuration:

- embedding model: sentence-transformers/all-MiniLM-L6-v2
- min_domain = 2
- max_domain = 10
- bootstrap = 5,000
- seed = 123

First full run result:

- eligible rows = 1,626
- surface_format:
  60.21%, CI [57.81%, 62.61%]
- top3 domain_contrast_sign_rule:
  88.07%, CI [86.47%, 89.61%]
- top5 domain_contrast_sign_rule:
  87.88%, CI [86.29%, 89.48%]
- mean domain_contrast_sign_rule:
  87.76%, CI [86.16%, 89.36%]
- top2 domain_contrast_sign_rule:
  87.52%, CI [85.92%, 89.11%]
- max domain_contrast_sign_rule:
  85.24%, CI [83.52%, 86.90%]

Interpretation:

This strongly supports the critic's operator-confound diagnosis. MiniLM embeddings can recover a strong chosen/rejected class-separation signal when the comparison operator is per-story/top-k rather than centroid cosine. Therefore, the earlier centroid-kernel null should not be interpreted as evidence for compression-specificity.

Important comparability caveat:

This first full embedding-operator run reports n=1,626, whereas the strongest compression V overlap reports n=1,155. We therefore need an exact-overlap rerun restricted to the row_ids from test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10 before making an apples-to-apples comparison.

## Exact-overlap crossed embedding operator control

We reran the crossed embedding operator control restricted to the exact row_ids from the strongest compression V run:

- row filter: test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10
- n = 1,155 pairs
- embedding model: sentence-transformers/all-MiniLM-L6-v2
- min_domain = 2
- max_domain = 10
- bootstrap = 5,000
- seed = 123

This directly tests the critic's concern that previous TF-IDF/MiniLM controls confounded representation with comparison operator. The earlier centroid controls used symmetric centroid cosine, whereas the compression-style probe used directional candidate-to-domain comparison.

### Main exact-overlap results

Compression V / conditional cross-predictability, max_domain=10:

- domain_contrast_sign_rule:
  79.74%, CI [77.40%, 82.08%]
- domain_specificity_logistic:
  79.57%, CI [77.23%, 81.90%]

MiniLM per-story/top-k operators on the same n=1,155 overlap:

- top3 + surface:
  88.66%, CI [86.84%, 90.39%]
- top2 + surface:
  88.48%, CI [86.58%, 90.30%]
- top5 + surface:
  88.40%, CI [86.49%, 90.22%]
- top3 domain_contrast_sign_rule:
  87.71%, CI [85.80%, 89.61%]
- top3 domain_specificity_logistic:
  87.71%, CI [85.80%, 89.61%]
- top5 domain_contrast_sign_rule:
  87.45%, CI [85.54%, 89.35%]
- mean domain_contrast_sign_rule:
  87.36%, CI [85.37%, 89.26%]
- top2 domain_contrast_sign_rule:
  87.10%, CI [85.11%, 89.00%]
- max domain_contrast_sign_rule:
  84.50%, CI [82.42%, 86.58%]

Surface baseline on same row set:

- surface_format:
  60.78%, CI [57.92%, 63.55%]

### Paired deltas over surface

- top3 sign/logistic - surface:
  +26.93 points, CI [+23.55,+30.30]
- top5 sign/logistic - surface:
  +26.67 points, CI [+23.29,+30.13]
- mean sign/logistic - surface:
  +26.58 points, CI [+23.20,+30.04]
- top2 sign/logistic - surface:
  +26.32 points, CI [+22.94,+29.70]
- max sign rule - surface:
  +23.72 points, CI [+20.26,+27.19]

### Continuous effects

MiniLM per-story/top-k operators show clean directional structure:

- top3 domain_contrast_delta:
  +0.3532, CI [+0.3356,+0.3705]
- top2 domain_contrast_delta:
  +0.4249, CI [+0.4045,+0.4452]
- top5 domain_contrast_delta:
  +0.2685, CI [+0.2545,+0.2824]
- mean domain_contrast_delta:
  +0.2110, CI [+0.1996,+0.2223]
- max domain_contrast_delta:
  +0.5167, CI [+0.4912,+0.5422]

For each operator, chosen_domain_specificity is positive and rejected_domain_specificity is negative, with bootstrap CIs excluding zero.

### Interpretation

The exact-overlap crossed operator control supports the critic's diagnosis.

The earlier MiniLM centroid null does not show that embeddings cannot recover the signal. It shows that centroid pooling destroys the signal. When MiniLM uses a directional per-story/top-k candidate-to-domain comparison, it recovers a very strong chosen/rejected class-separation signal and exceeds the conditional LM cross-predictability result on the exact same row set.

Updated conclusion:

> The durable finding is not compression-specificity. It is directional supervised class separability. Preference-labeled chosen/rejected domains form a strong same-prompt class boundary, and directional candidate-to-domain comparison detects that boundary. Centroid pooling obscures it.

Updated terminology:

Use "conditional cross-predictability" for the LM operational score. Avoid implying that the current evidence establishes compression-progress V as uniquely mechanizing human value.

Remaining decisive open problem:

The domains are still label-defined. The next key test is label-leakage-reduced domain construction.

## Exact-overlap crossed embedding operator control

We reran the crossed embedding operator control restricted to the exact row_ids from the strongest compression V run:

- row filter: test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10
- n = 1,155 pairs
- embedding model: sentence-transformers/all-MiniLM-L6-v2
- min_domain = 2
- max_domain = 10
- bootstrap = 5,000
- seed = 123

This directly tests the critic's concern that previous TF-IDF/MiniLM controls confounded representation with comparison operator. The earlier centroid controls used symmetric centroid cosine, whereas the compression-style probe used directional candidate-to-domain comparison.

### Main exact-overlap results

Compression V / conditional cross-predictability, max_domain=10:

- domain_contrast_sign_rule:
  79.74%, CI [77.40%, 82.08%]
- domain_specificity_logistic:
  79.57%, CI [77.23%, 81.90%]

MiniLM per-story/top-k operators on the same n=1,155 overlap:

- top3 + surface:
  88.66%, CI [86.84%, 90.39%]
- top2 + surface:
  88.48%, CI [86.58%, 90.30%]
- top5 + surface:
  88.40%, CI [86.49%, 90.22%]
- top3 domain_contrast_sign_rule:
  87.71%, CI [85.80%, 89.61%]
- top3 domain_specificity_logistic:
  87.71%, CI [85.80%, 89.61%]
- top5 domain_contrast_sign_rule:
  87.45%, CI [85.54%, 89.35%]
- mean domain_contrast_sign_rule:
  87.36%, CI [85.37%, 89.26%]
- top2 domain_contrast_sign_rule:
  87.10%, CI [85.11%, 89.00%]
- max domain_contrast_sign_rule:
  84.50%, CI [82.42%, 86.58%]

Surface baseline on same row set:

- surface_format:
  60.78%, CI [57.92%, 63.55%]

### Paired deltas over surface

- top3 sign/logistic - surface:
  +26.93 points, CI [+23.55,+30.30]
- top5 sign/logistic - surface:
  +26.67 points, CI [+23.29,+30.13]
- mean sign/logistic - surface:
  +26.58 points, CI [+23.20,+30.04]
- top2 sign/logistic - surface:
  +26.32 points, CI [+22.94,+29.70]
- max sign rule - surface:
  +23.72 points, CI [+20.26,+27.19]

### Continuous effects

MiniLM per-story/top-k operators show clean directional structure:

- top3 domain_contrast_delta:
  +0.3532, CI [+0.3356,+0.3705]
- top2 domain_contrast_delta:
  +0.4249, CI [+0.4045,+0.4452]
- top5 domain_contrast_delta:
  +0.2685, CI [+0.2545,+0.2824]
- mean domain_contrast_delta:
  +0.2110, CI [+0.1996,+0.2223]
- max domain_contrast_delta:
  +0.5167, CI [+0.4912,+0.5422]

For each operator, chosen_domain_specificity is positive and rejected_domain_specificity is negative, with bootstrap CIs excluding zero.

### Interpretation

The exact-overlap crossed operator control supports the critic's diagnosis.

The earlier MiniLM centroid null does not show that embeddings cannot recover the signal. It shows that centroid pooling destroys the signal. When MiniLM uses a directional per-story/top-k candidate-to-domain comparison, it recovers a very strong chosen/rejected class-separation signal and exceeds the conditional LM cross-predictability result on the exact same row set.

Updated conclusion:

> The durable finding is not compression-specificity. It is directional supervised class separability. Preference-labeled chosen/rejected domains form a strong same-prompt class boundary, and directional candidate-to-domain comparison detects that boundary. Centroid pooling obscures it.

Updated terminology:

Use "conditional cross-predictability" for the LM operational score. Avoid implying that the current evidence establishes compression-progress V as uniquely mechanizing human value.

Remaining decisive open problem:

The domains are still label-defined. The next key test is label-leakage-reduced domain construction.

## Next experiment: label-leakage-reduced training-domain probe

The exact-overlap crossed embedding operator control showed that MiniLM embeddings recover a strong chosen/rejected class-separation signal when using directional per-story/top-k comparison rather than centroid cosine. This weakens compression-specificity and supports the critic's operator-confound diagnosis.

The remaining decisive issue is label leakage:

- Previous same-prompt domains used test labels to construct D_preferred and D_rejected.
- That is supervised class probing, not label-free value discovery.

Next experiment:

Build preferred/rejected domains from the LitBench training split, retrieved by prompt similarity, then evaluate on held-out test pairs.

For each test prompt:

1. Embed the test prompt.
2. Retrieve nearest training prompts.
3. Build D_preferred from chosen stories in the retrieved training pairs.
4. Build D_rejected from rejected stories in the retrieved training pairs.
5. Score the held-out test chosen/rejected pair using MiniLM directional per-story/top-k operators.
6. Evaluate only at the end against test labels.

This removes direct test-label domain construction. It remains supervised because training labels define the domains, but it is label-leakage-reduced relative to same-test-prompt leave-pair-out domains.

Decision rule:

- If performance remains high, the result becomes a supervised train-to-test selector/prototype probe rather than a test-label leakage artifact.
- If performance collapses toward surface/chance, the previous 87–88% result mainly reflected test-set label-defined domain construction.

## Label-leakage-reduced train-domain probe

We tested whether the strong same-test label-defined domain result transfers when domains are built from training labels rather than test labels.

Setup:

- train cache: results/litbench/litbench_train_cached.csv
- train pairs: 43,827
- test dataset: SAA-Lab/LitBench-Test-IDs-Complete
- test pairs: 2,480
- embedding model: sentence-transformers/all-MiniLM-L6-v2
- retrieval: nearest training prompts by MiniLM prompt embedding
- D_preferred: chosen stories from retrieved training pairs
- D_rejected: rejected stories from retrieved training pairs
- top_train_pairs = 50
- max_domain = 10
- min_domain = 2
- n_boot = 5,000
- seed = 123

This is label-leakage-reduced relative to the previous same-test leave-pair-out domains because test labels are not used to build D_preferred or D_rejected.

### Main results

Surface baseline:

- surface_format:
  60.16%, CI [58.15%, 62.14%]

Best train-domain-only results:

- top5_train_domain_sign_rule/logistic:
  53.55%, CI [51.61%, 55.52%]
- mean_train_domain_sign_rule/logistic:
  53.47%, CI [51.45%, 55.36%]
- top2_train_domain_sign_rule/logistic:
  53.19%, CI [51.17%, 55.12%]
- top3_train_domain_sign_rule/logistic:
  52.98%, CI [51.05%, 54.96%]
- max_train_domain_sign_rule/logistic:
  52.02%, CI [50.04%, 53.99%]

Surface + train-domain:

- top5_surface_plus_train_domain:
  60.28%, CI [58.27%, 62.22%]
- top2_surface_plus_train_domain:
  60.28%, CI [58.31%, 62.22%]
- mean_surface_plus_train_domain:
  60.20%, CI [58.23%, 62.18%]
- top3_surface_plus_train_domain:
  60.12%, CI [58.15%, 62.06%]
- max_surface_plus_train_domain:
  60.04%, CI [58.06%, 61.98%]

Paired interpretation:

The train-domain feature adds essentially no resolved value over surface. Surface + train-domain is within roughly ±1 point of surface, with CIs crossing zero.

Continuous effects:

The train-domain directional effect is technically positive but tiny:

- mean domain_contrast_delta: +0.0039
- max domain_contrast_delta: +0.0054
- top2 domain_contrast_delta: +0.0052
- top3 domain_contrast_delta: +0.0053
- top5 domain_contrast_delta: +0.0047

Interpretation:

The leakage-reduced train-domain probe mostly collapses. This indicates that the very large same-test domain result depends heavily on label-defined test-domain construction. The same-test top-k MiniLM result should therefore be framed as supervised label-class probing, not as label-free value discovery or transferable value selection.

Updated conclusion:

> The durable finding is that preference-labeled domains create strong same-test class boundaries that directional candidate-to-domain operators can detect. But when domains are built from training labels and transferred to held-out test examples, the signal is weak and adds no resolved value beyond formatting.

This supports the negative-with-relocation interpretation:

> The human label signal is carried by D. The selector frontier remains human; the engineering leverage is in constructing useful domains, not removing human judgment.

## Exact-overlap train-domain leakage-reduced probe

We reran the train-domain leakage-reduced probe on the exact 1,155 rows from the strongest same-test domain contrast run:

- row filter: test_prompt_v_distilgpt2_domaincontrast_mindomain2_maxdomain10
- train pairs: 43,827
- test rows: 1,155
- domain source: nearest training prompts by MiniLM prompt embedding
- D_preferred: chosen stories from retrieved training pairs
- D_rejected: rejected stories from retrieved training pairs
- embedding model: sentence-transformers/all-MiniLM-L6-v2
- top_train_pairs = 50
- max_domain = 10
- min_domain = 2
- n_boot = 5,000
- seed = 123

### Main comparison

Same-test MiniLM top-k domain probe on exact 1,155:

- top3 domain_contrast_sign_rule:
  87.71%, CI [85.80%, 89.61%]
- top5 domain_contrast_sign_rule:
  87.45%, CI [85.54%, 89.35%]
- mean domain_contrast_sign_rule:
  87.36%, CI [85.37%, 89.26%]

Train-domain leakage-reduced probe on exact 1,155:

- top5_train_domain_sign_rule/logistic:
  53.07%, CI [50.22%, 55.93%]
- mean_train_domain_sign_rule/logistic:
  52.55%, CI [49.70%, 55.41%]
- top2_train_domain_sign_rule/logistic:
  51.77%, CI [48.83%, 54.63%]
- top3_train_domain_sign_rule/logistic:
  51.52%, CI [48.66%, 54.37%]
- max_train_domain_sign_rule/logistic:
  50.65%, CI [47.79%, 53.59%]

Surface baseline on exact 1,155:

- surface_format:
  60.78%, CI [57.92%, 63.55%]

Surface + train-domain:

- top3_surface_plus_train_domain:
  61.30%, CI [58.53%, 64.07%]
- top5_surface_plus_train_domain:
  61.13%, CI [58.35%, 63.90%]
- top2_surface_plus_train_domain:
  61.04%, CI [58.27%, 63.81%]
- max_surface_plus_train_domain:
  60.95%, CI [58.10%, 63.72%]
- mean_surface_plus_train_domain:
  60.61%, CI [57.75%, 63.38%]

Paired deltas over surface:

- top3_surface_plus_train_domain - surface:
  +0.52 points, CI [-0.87, +1.90]
- top5_surface_plus_train_domain - surface:
  +0.35 points, CI [-1.04, +1.73]
- top2_surface_plus_train_domain - surface:
  +0.26 points, CI [-1.04, +1.47]
- max_surface_plus_train_domain - surface:
  +0.17 points, CI [-1.21, +1.47]
- mean_surface_plus_train_domain - surface:
  -0.17 points, CI [-1.90, +1.47]

### Interpretation

The exact-overlap result confirms that the full-set train-domain collapse was not merely population dilution.

On the same 1,155 rows where same-test MiniLM top-k domains reached roughly 87–88%, train-built domains reach only roughly 50.6–53.1% alone and add no resolved value beyond surface formatting.

Conclusion:

> The large same-test result is best interpreted as supervised label-domain probing. Directional operators detect class structure when D is built from the same test-label ecology, but train-built domains do not transfer that structure meaningfully to held-out test labels.

Updated paper-level framing:

> The human signal is carried by domain construction. The selector frontier remains human; the engineering leverage is in constructing useful D, not eliminating human judgment.

## Next missing cell: different-prompt test-label domains

The critic pointed out that the leakage-reduced train-domain probe changed two factors at once:

1. label source:
   same-test labels -> training labels

2. prompt match:
   same prompt -> nearest different prompts by embedding similarity

Therefore the collapse of train-built domains could reflect either:

- removal of same-test label ecology; or
- poor / insufficient prompt matching when domains are drawn from different prompts.

The missing cell is:

- score test examples using domains built from other test prompts' chosen/rejected stories;
- exclude the same prompt;
- retrieve nearest different test prompts by MiniLM prompt similarity;
- build D_preferred from chosen stories of those retrieved test prompts;
- build D_rejected from rejected stories of those retrieved test prompts.

This is still transductive and uses test labels, but removes the literal same-prompt domain ecology.

Interpretation:

- If different-prompt test-label domains collapse near the train-domain result, the mechanism is prompt-local label ecology.
- If they stay high, then test-label source rather than prompt-locality is doing more work.
- If they land in the middle, the boundary is partly cross-prompt but much weaker than same-prompt transductive domains.
