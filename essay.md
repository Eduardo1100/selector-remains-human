# The selector remains human

Creative systems can generate an abundance of plausible work. The harder
question is which work to keep. This project began with a tempting answer:
perhaps a mechanical compression signal could become a selector, finding value
without needing people to state what they value.

The result is a useful no.

On human-rated poetry, generic literary compression did not recover appraisal
and could point in the wrong direction. On LitBench, a strong result did
appear—but only when each candidate was compared against other
preference-labeled responses to the *same prompt*. That is not a free-standing
creative judge. It is a transductive readout of a local human-made boundary.

The mechanism became clearer when we changed the tool. A directional MiniLM
comparison beat the conditional language-model score on the same 1,155
examples. A centroid representation failed. Randomizing labels within a prompt
failed. Moving labeled examples to different prompts failed, even when those
labels came from the test set. The apparent miracle was neither compression nor
prompt similarity alone. It was prompt-local human label ecology.

![Horizontal point-and-interval plot: same-prompt preference labels create a strong transductive boundary, while random same-prompt domains and train- or test-labeled cross-prompt domains remain near chance; vertical reference lines mark chance and the surface-format baseline.](paper/figures/selector_remains_human_figure1b.svg)

That is not bad news for creative tools. It says where to put the engineering
effort: help people curate exemplars, preserve context, express disagreement,
and maintain living comparison sets. Machines can make those domains legible
and scalable. But they do not make the human source of value disappear.

The selector remains human.
