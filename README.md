# Anonymized Code for the Model Organism Lottery

This repository holds an anonymized version of the codebase from the paper "The Model Organism Lottery: Model Organism Interpretability Strongly Depends on Training Methodology".

## Layout

The main repository is `model-organism-lottery`, represented here as the `model-organism-lottery-e773d2d58e3ad556356625df31e38f253cfac428` snapshot. That repository contains the following Git submodules, represented here as sibling snapshots:

  * submodule `diffing-toolkit` -> snapshot `diffing-toolkit-ac5595d277bf307b23e32f0953a32d8154d7c152` (our fork of https://github.com/science-of-finetuning/diffing-toolkit)
  * submodule `SAELens` -> snapshot `SAELens-58bd29c7d7b1ad39cbf09cee2d72fc5c5234b8e2` (our fork of https://github.com/decoderesearch/SAELens)
  * submodule `open-instruct-1b` -> snapshot `open-instruct-1b-7220ca8627323c5b7d830a6caf1c382605a86845` (our fork of https://github.com/allenai/open-instruct)
  * submodule `alpaca_eval` -> snapshot `alpaca_eval-472a9ddc8fc99d463cff6abeb58e17b0ab285b7e` (our fork of https://github.com/tatsu-lab/alpaca_eval)

`model-organism-lottery` also contains a `lm-evaluation-harness` submodule pointing directly to the upstream https://github.com/EleutherAI/lm-evaluation-harness/tree/95d580638385578c1c07fa554cf16ad7f5b5f460.

As described more fully in `diffing-toolkit-ac5595d277bf307b23e32f0953a32d8154d7c152/README.md`, certain results use a slightly different version of certain `diffing-toolkit` files, represented here as partial snapshot `diffing-toolkit-c3c8dc8d2ca2583651afe7fe6ae3339a9dc3c8b8`.
