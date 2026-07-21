# The Selector Remains Human
[Project site](https://eduardo1100.github.io/poemforge-paper/) ·
[Paper PDF](https://eduardo1100.github.io/poemforge-paper/paper/main.pdf) ·
[Release v0.1.0](https://github.com/Eduardo1100/poemforge-paper/releases/tag/v0.1.0)
## A negative result on compression-based value signals for creative text

This repository accompanies a working paper by Eduardo Cortes. It asks whether
mechanical, compression-based readouts can replace human creative appraisal.
They do not. The strongest LitBench signal appears only when comparison domains
are built from other **preference-labeled responses to the same prompt**. A
directional MiniLM control is stronger than the conditional language-model
readout on the same rows. When labels are randomized or domains move to other
prompts, the effect collapses.

> The selector remains human. The useful engineering work is helping people
> construct and maintain comparison domains, not removing them from the loop.

The paper is a negative-with-relocation result, not a claim of a deployable
creative evaluator. Canonical Figure 1 numbers are generated from tracked
result summaries and recorded in its provenance manifest.

![Horizontal point-and-interval plot: same-prompt preference labels achieve a strong transductive boundary, while random same-prompt domains and train- or test-labeled cross-prompt domains remain near chance; dashed and dotted vertical lines mark chance and the surface baseline.](paper/figures/selector_remains_human_figure1b.svg)

## Read the work

- [Working manuscript](paper/main.md)
- [Download the working-paper PDF](paper/main.pdf)
- [Standalone public essay](essay.md)
- [Primary-figure data and provenance](paper/figures/figure1_data.json)
- [Reproducibility guide](docs/reproducibility.md)
- [Publication bundle checklist](docs/PUBLICATION.md)
- [Self-contained static bundle source](site/index.html)

## Build safely

    make paper-assets
    make site
    make publication-check

Make reproduce remains the backward-compatible Phase A pipeline. LitBench
recomputation is explicit and separately documented; neither paper nor site
builds download models or run inference.

## License scope

Repository software is available under the [MIT License](LICENSE). Original
manuscript, documentation, essay, site prose, and authored tables and figures
are available under [CC BY 4.0](LICENSE-CONTENT.md). Third-party datasets,
frozen external artifacts, model outputs, and material carrying its own terms
are excluded from both grants; see [LICENSE-CONTENT.md](LICENSE-CONTENT.md).
