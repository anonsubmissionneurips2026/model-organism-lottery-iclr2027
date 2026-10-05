# Anonymized Code for the Model Organism Lottery

This repository holds an anonymized version of the codebase from the paper "The Model Organism Lottery: Model Organism Interpretability Strongly Depends on Training Methodology".

## Layout

The main repository is `model-organism-lottery`, represented here as the `model-organism-lottery-ac04e6039055828df78a1aeb3b2c01e822f22faa` snapshot. That repository contains the following Git submodules, represented here as sibling snapshots:

  * submodule `diffing-toolkit` -> snapshot `diffing-toolkit-0caf69e1108e36c97461399bd5732580356e4db0` (our fork of https://github.com/science-of-finetuning/diffing-toolkit)
  * submodule `SAELens` -> snapshot `SAELens-58bd29c7d7b1ad39cbf09cee2d72fc5c5234b8e2` (our fork of https://github.com/decoderesearch/SAELens)
  * submodule `open-instruct-1b` -> snapshot `open-instruct-1b-7220ca8627323c5b7d830a6caf1c382605a86845` (our fork of https://github.com/allenai/open-instruct)
  * submodule `alpaca_eval` -> snapshot `alpaca_eval-472a9ddc8fc99d463cff6abeb58e17b0ab285b7e` (our fork of https://github.com/tatsu-lab/alpaca_eval)
  * submodule `jacobian-lens` -> snapshot `jacobian-lens-3d772c5da8c5f54bce8eb70a9658ea7ce385c536` (our fork of https://github.com/anthropics/jacobian-lens)

Separately, we use our `auto-mo` repository, represented here as the `auto-mo-6ee2c9926eefe596a8292b1daa7e2f10d19e236d` snapshot.

`model-organism-lottery` also contains a `lm-evaluation-harness` submodule pointing directly to the upstream https://github.com/EleutherAI/lm-evaluation-harness/tree/95d580638385578c1c07fa554cf16ad7f5b5f460.
