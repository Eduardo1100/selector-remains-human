# Reproducibility

## Safe defaults

Make reproduce and make reproduce-phase-a run only the established Phase A
pipeline. They do not include LitBench. Publication commands read existing
tracked result artifacts only:

    make paper-assets
    make site
    make publication-check

paper/main.md is the editorial source. Pandoc generates paper/main.tex.
PDF output can use Tectonic when present, or latexmk with a PDF-capable TeX
installation.

The repository-local Mise toolchain pins Pandoc 3.10 and Tectonic 0.16.9.
Run mise install to provision document tools, then run mise exec -- make paper
to build the PDF. mise.toml is tracked source configuration, not generated
state.

## LitBench recomputation tiers

These targets are explicit because they require remote LitBench data, optional
dependencies, local caches, and potentially substantial inference time.

| Target | Stages | Inputs and cost |
|---|---|---|
| reproduce-litbench-inventory | 00, 04 | Downloads LitBench splits through datasets; inexpensive inventory work. |
| reproduce-litbench-surface | 01 | Downloads test data; surface-feature baseline and cross-validation. |
| reproduce-litbench-lm | 02, 03, 05, 06, 07, 08 | Downloads LitBench data and DistilGPT-2 when uncached; recomputes NLL and conditional LM scores; expensive. |
| reproduce-litbench-embedding | 09, 12, 13, 14 | Downloads data and MiniLM when uncached; recomputes sentence embeddings and domain probes; moderately expensive to expensive. |
| reproduce-litbench-reward | 10 | Downloads data and a sequence-classification model when uncached; optional, expensive inference. |
| reproduce-litbench-summary | 11 | Reads results from the other tiers; inexpensive, but cannot run until its inputs exist. |
| reproduce-litbench-full | all tiers | Full remote-data/model recreation; expensive and not a default. |

The optional LitBench dependencies are not declared in the base project:
datasets, pyarrow, torch, transformers, and sentence-transformers.
The train-domain probe can read a local cache, but its default cache is not a
portable tracked input; a clean clone therefore needs access to LitBench data.

## Canonical Figure 1 provenance

Run python scripts/make_primary_figure.py --check to validate every selected
row, then run it without --check to write the committed figure-data manifest
and assets. The manifest records source file, filters, accuracy, interval,
sample size, operator, label source, prompt relation, and transductive or
inductive status.

The surface-format baseline is intentionally selected from:

    results/litbench/litbench_prompt_domain_embedding_operator_models_test_embedding_operators_exact1155_mindomain2_maxdomain10.csv

with operator=mean and model=surface_format: 0.607792, n=1,155. The
conditional-LM result summary has a 0.602597 surface row because each script
runs its own five-fold cross-validation prediction procedure. Those prediction
sets are separate result artifacts despite the shared nominal sample size.
Figure 1 and the manuscript use the exact-overlap embedding-summary row because
it is the canonical source specified for the row-identical operator comparison.

## Licensing

No license has been chosen. Releasing a public bundle requires an owner
decision on licensing first.
