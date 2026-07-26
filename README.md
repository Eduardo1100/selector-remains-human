# The Selector Remains Human

[Project site](https://eduardo1100.github.io/selector-remains-human/) ·
[Paper PDF](https://eduardo1100.github.io/selector-remains-human/paper/main.pdf) ·
[Founder essay](essay.md) ·
[Idea Search artifact](https://eduardo1100.github.io/manifold-studio/) ·
[Release v0.1.0](https://github.com/Eduardo1100/selector-remains-human/releases/tag/v0.1.0)

## The 30-second version

I built this project after an AI-native media pipeline made the real bottleneck
obvious: models could generate more songs, scenes, and creative directions than
I could meaningfully choose among. I wanted to know whether a mechanical signal
could supply the missing judgment.

Across a human-rated poetry corpus and LitBench, the generic proxies I tested
did not replace appraisal. A strong boundary did appear when candidates were
compared with **preference-labeled responses to the same prompt**, but crossed
controls showed that the signal came from the local human-shaped comparison
set—not from compression itself and not from a transferable value function.

> The result is negative with relocation: human judgment can move out of
> per-candidate scoring and into reference-set construction, frontier
> decisions, commitments, and drift checks. It did not disappear.

This finding shaped Manifold's architecture. Generators supply possibilities;
creators define decision rights; audiences compare concrete alternatives; and
the platform records commitments, fulfillment, and the resulting property
history. The research motivates that design. It does **not** validate consumer
demand for Manifold.

![Horizontal point-and-interval plot: same-prompt preference labels achieve a strong transductive boundary, while random same-prompt domains and train- or test-labeled cross-prompt domains remain near chance; dashed and dotted vertical lines mark chance and the surface baseline.](paper/figures/selector_remains_human_figure1b.svg)

## What changed my mind

| Question | Decisive result | What it ruled out |
|---|---:|---|
| Can generic compression act as a label-free creative selector? | No resolved positive alignment on poetry; matched controls could anti-align | Compression as a sufficient general proxy for appraisal |
| Was the strong LitBench result compression-specific? | Directional MiniLM: **87.71%** vs conditional LM: **79.74%** on the same 1,155 rows | The interpretation that compression was the unique mechanism |
| Was sharing the same prompt sufficient? | Random same-prompt domains: **48.92%** | Prompt similarity without preference labels |
| Could a pooled prototype preserve the boundary? | MiniLM centroid: **48.92%** | Taste as one averaged prototype |
| Did human-shaped comparison domains transfer? | Cross-prompt train/test label pools: **≈53%** | A general creative evaluator learned from another prompt ecology |

The 87.71% result is a **transductive probe**: its reference domains contain
labels from other held-out test items. It is evidence that human labels induce
a strong local boundary, not a reward-model benchmark and not a deployable
judge.

## Founder path

1. **AI-native media pipeline.** I built a reusable production system spanning
   research, music, image and video synthesis, editing, captioning, and release
   packaging. Generation became cheap; selection became the constraint.
2. **Manifold Studio / Idea Search.** I separated proposal from judgment using
   concrete candidates, pairwise choices, remembered constraints, and mutation.
   The public [Idea Search artifact](https://eduardo1100.github.io/manifold-studio/)
   exposes the method without exposing the private application.
3. **Selector research.** I tested compression progress, model surprise,
   conditional-LM readouts, TF-IDF and MiniLM domains, centroid and directional
   operators, random-label controls, prompt transfer, surface baselines, and
   bootstrap uncertainty.
4. **Manifold.** The product leap was to move selection from a private creator
   tool into a social system: bounded audience decisions, explicit authority,
   creator commitments, fulfillment, and visible lineage.

This is the intellectual history of the product, not a claim that the research
predicted the market.

## What is in this repository

| Layer | Concrete contents |
|---|---|
| Data and provenance | Processed human ratings and pairwise comparisons, input manifests, frozen-result inventories, and machine-readable figure provenance |
| Mechanisms | Compression and surprise readouts, conditional language-model domains, TF-IDF and MiniLM controls, centroid and directional instance-level operators |
| Falsification | Identical-row operator crosses, random same-prompt labels, cross-prompt train/test pools, surface baselines, and bootstrap intervals |
| Publication | Generated tables and figures, working manuscript and PDF, public founder essay, reproducibility documentation, licenses, and a self-contained static site |

## Read and inspect

- [Working manuscript](paper/main.md)
- [Download the working-paper PDF](paper/main.pdf)
- [Why Manifold begins after the generator](essay.md)
- [Primary-figure data and provenance](paper/figures/figure1_data.json)
- [Reproducibility guide](docs/reproducibility.md)
- [Publication bundle checklist](docs/PUBLICATION.md)
- [Self-contained static bundle source](site/index.html)
- [Experiment plans and interpretation history](docs/experiment_notes/)

## Reproduce the publication artifact

```bash
make paper-assets
make site
make publication-check
```

`make reproduce` remains the backward-compatible Phase A pipeline. LitBench
recomputation is explicit and separately documented; neither paper nor site
build downloads models or runs inference. The publication checks verify
provenance-selected numbers, bundle completeness, links, citation metadata, and
license notices.

## Evidence boundary

Established here:

- The tested generic compression and surprise proxies were insufficient.
- The strongest LitBench boundary depended on prompt-local human preference
  labels and directional per-instance comparison.
- Changing the operator could erase or recover the signal on the same rows.
- The preference-labeled boundary did not transfer across prompt ecologies.

Not established here:

- That no possible mechanical or world-grounded value signal can exist.
- That every creative domain requires identical human involvement.
- That the research validates Manifold's acquisition, retention, or
  monetization hypotheses.
- That the 87.71% transductive result is comparable to an inductive reward
  model.

## License scope

Repository software is available under the [MIT License](LICENSE). Original
manuscript, documentation, essay, site prose, and authored tables and figures
are available under [CC BY 4.0](LICENSE-CONTENT.md). Third-party datasets,
frozen external artifacts, model outputs, and material carrying its own terms
are excluded from both grants; see [LICENSE-CONTENT.md](LICENSE-CONTENT.md).
