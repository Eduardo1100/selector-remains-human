# Why Manifold begins after the generator

*Eduardo Cortes · July 2026*

## The problem I encountered

I built an AI-native production pipeline for short films and music releases.
It covered research, candidate generation, music, images, video, editing,
captioning, and release packaging. The pipeline made one part of creative work
radically cheaper: producing another plausible option.

It did not make the next decision cheaper.

Once I could generate many songs, shots, visual treatments, hooks, and
narrative directions, the bottleneck moved. The hard question was no longer
"Can the model make something?" It was "Which possibility deserves the next
hour of work, and why?"

That distinction led me to a generator-selector model. The generator proposes.
The selector supplies value-ordering from a source whose authority does not
come from the generator itself. A model's probability distribution is excellent
at producing likely continuations; using the same distribution as the final
judge risks confusing typicality with value.

## The first system

I built Manifold Studio to separate those roles in practice. Instead of asking
for one output and accepting it, I could create candidates, compare them,
preserve constraints, record "more like this" and "less like this," and mutate
the surviving direction. The public
[Idea Search artifact](https://eduardo1100.github.io/manifold-studio/) presents
that method without exposing the private application.

The system still depended on judgment. That raised a more ambitious question:
could I mechanize the selector without quietly teaching it yesterday's labels?

## The shortcut I hoped would work

Compression was a plausible candidate. A useful artifact might improve an
observer's model of its domain: familiar enough to be legible, novel enough to
change prediction. Compression progress, model surprise, or a related
information-theoretic readout could in principle remain responsive to new work
without requiring a person to score every candidate.

I tested that idea first on human-rated poetry and then on LitBench, a
pairwise-preference benchmark for creative writing. The repository implements
conditional language-model domains, generic compression and surprise readouts,
surface controls, TF-IDF and MiniLM comparison domains, centroid and
directional per-instance operators, random-label controls, cross-prompt
transfer, and bootstrap uncertainty.

## What the experiments changed

The generic proxies failed. On poetry, literary compression did not recover
human appraisal and could point in the wrong direction under matched controls.
Higher-rated poems were not simply the ones that surprised the model more.

LitBench produced a more informative result. A conditional language-model
readout reached 79.7% pairwise accuracy when each candidate was evaluated
against other preference-labeled responses to the same prompt. Initially, that
looked like evidence for a compression-based selector.

Then the crossed controls changed the interpretation:

- A directional MiniLM top-3 operator reached **87.7% on the same 1,155 rows**.
  Compression was not the unique mechanism.
- MiniLM centroid pooling fell to **48.9%**. Averaging the reference set erased
  a boundary that specific candidate-to-exemplar comparisons preserved.
- Random same-prompt domains also fell to **48.9%**. Sharing a prompt was not
  enough; the human preference labels were load-bearing.
- Preference-labeled domains moved to other prompts stayed near **53%**,
  whether the labels came from train or test items. The boundary did not become
  a general creative evaluator.

![Horizontal point-and-interval plot: same-prompt preference labels create a strong transductive boundary, while random same-prompt domains and train- or test-labeled cross-prompt domains remain near chance; vertical reference lines mark chance and the surface-format baseline.](paper/figures/selector_remains_human_figure1b.svg)

The 87.7% number is a transductive probe. Its domains contain labels from other
held-out test items. It is evidence that human labels create a strong local
class boundary, not a reward-model benchmark and not a deployable judge.

The useful conclusion was narrower and more actionable: in these experiments,
the construction of the comparison set mattered more than the readout family,
specific comparisons mattered more than a pooled prototype, and value
structure did not travel cleanly across local ecologies.

## Where the human work moved

This does not imply that a person must approve every candidate. Most of the
production loop can remain machine work: generation, diversity, revision
execution, cheap readout over a useful comparison set, and bulk low-novelty
scoring.

The human contribution moves upward:

1. **Construct the reference set.** Choose the exemplars and contrasts that
   constitute the relevant local taste.
2. **Adjudicate the frontier.** Intervene where novelty makes a frozen proxy
   least reliable.
3. **Make high-leverage commitments.** Set premises and path-dependent
   directions that downstream execution inherits.
4. **Certify non-drift.** Check that the machine's ordering still resembles the
   intended human one.

An agent can still generate, revise, publish, interact, and operate a persistent
creative property. The claim is not that agents cannot create. It is that
automating creative labor does not automatically provide an independent reason
for what a creator or culture should value.

## The social leap

Manifold is not a better scoring model. It is the social system around
selection.

Creators define bounded decisions: what is open to influence, which candidates
are legitimate, how the outcome resolves, whether it is binding, what approval
authority remains, and what fulfillment will mean. Audiences compare concrete
possibilities. A Loop Contract records the commitment, decision, fulfillment
state, and descendant release. The result becomes part of the property's
history and context for the next decision.

That architecture follows three principles from the research:

- **Concrete comparison:** judge realized alternatives, not an abstract intent.
- **Local authority:** a community decision establishes what this community
  prefers under this contract; it does not discover universal artistic quality.
- **Visible consequence:** selection becomes meaningful when people can see
  what their action changed.

This also explains why Manifold begins after the generator. Generation models
will change. The property, its decision rights, audience relationship,
commitments, fulfillment record, and cross-release lineage should persist.

## What remains unproven

The research shaped the product architecture. It did not validate the market.

It does not establish that audiences will join a dedicated AI-entertainment
network, return to see consequences, pay for participation, or prefer Manifold
to Twitch, YouTube, or another incumbent. It does not establish that creators
will accept public commitments or that asynchronous audience decisions will
retain the energy of live interaction.

Those are product questions and should be tested as such. The research answers
a different question: if generation is abundant, where should the system place
authority and judgment?

My answer changed from "inside a better mechanical score" to "inside explicit,
living relationships among creators, audiences, agents, and the history they
produce together."

The selector remains human—but the selection process can become a medium.
