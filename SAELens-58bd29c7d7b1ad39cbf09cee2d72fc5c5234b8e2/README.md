# Fork: Model Organism Interp

This is a research fork of [decoderesearch/SAELens](https://github.com/decoderesearch/SAELens),
repurposed as the `model-organism-interp` project. It uses the upstream SAELens
library as a foundation and adds an analysis pipeline for studying how fine-tuning
shifts SAE feature usage in quirked model organisms (e.g. Gemma 3 1B IT variants
trained to talk about submarines in military contexts, or to express Italian-food
preferences).

What this fork adds on top of upstream:

- `scripts/model_organism_interp_analysis/` — full pipeline (per-MO feature analysis,
  sibling-diff variants, LLM-based feature judging via OpenRouter, cross-MO noise
  floors, paper plots). See
  [`scripts/model_organism_interp_analysis/README.md`](scripts/model_organism_interp_analysis/README.md)
  for the full overview and command reference.
- uv-based dependency management (`uv.lock`, `pyproject.toml` rewrite) with a CUDA
  symlink fix-up script at `scripts/fix_cuda_libs.sh`.
- `CLAUDE.md` — project guidance for Claude Code.

## Setup

Run once after `uv sync`, or on a new machine:

```bash
uv sync
bash scripts/fix_cuda_libs.sh   # symlinks system CUDA .so files into the venv
```

## Auth

Required before running analysis. Gemma 3 is gated; OpenRouter is needed for the
LLM judge. Copy `.env.example` to `.env` and fill in both keys:

```
HF_TOKEN=<your_token>
OPENROUTER_API_KEY=<your_key>
```

## Run the full pipeline

```bash
bash scripts/model_organism_interp_analysis/run_binary_pipeline.sh
```

For per-step commands, judge options, regeneration flags, sibling pipeline, plots,
and exports, see
[`scripts/model_organism_interp_analysis/README.md`](scripts/model_organism_interp_analysis/README.md).

## Serve results locally

```bash
python3 -m http.server 8080 --directory results
# then open e.g.: http://localhost:8080/military_submarine_binary/runs/<run>_feature_analysis.html
#                 http://localhost:8080/italian_food_binary/runs/<run>_feature_analysis.html
```

---

The upstream SAELens README follows below, describing the underlying library.

---

<img width="1308" height="532" alt="saes_pic" src="https://github.com/user-attachments/assets/2a5d752f-b261-4ee4-ad5d-ebf282321371" />

# SAE Lens

[![PyPI](https://img.shields.io/pypi/v/sae-lens?color=blue)](https://pypi.org/project/sae-lens/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![build](https://github.com/decoderesearch/SAELens/actions/workflows/build.yml/badge.svg)](https://github.com/decoderesearch/SAELens/actions/workflows/build.yml)
[![Deploy Docs](https://github.com/decoderesearch/SAELens/actions/workflows/deploy_docs.yml/badge.svg)](https://github.com/decoderesearch/SAELens/actions/workflows/deploy_docs.yml)
[![codecov](https://codecov.io/gh/decoderesearch/SAELens/graph/badge.svg?token=N83NGH8CGE)](https://codecov.io/gh/decoderesearch/SAELens)

SAELens exists to help researchers:

- Train sparse autoencoders.
- Analyse sparse autoencoders / research mechanistic interpretability.
- Generate insights which make it easier to create safe and aligned AI systems.

SAELens inference works with any PyTorch-based model, not just TransformerLens. While we provide deep integration with TransformerLens via `HookedSAETransformer`, SAEs can be used with Hugging Face Transformers, NNsight, or any other framework by extracting activations and passing them to the SAE's `encode()` and `decode()` methods.

Please refer to the [documentation](https://decoderesearch.github.io/SAELens/) for information on how to:

- Download and Analyse pre-trained sparse autoencoders.
- Train your own sparse autoencoders.
- Generate feature dashboards with the [SAE-Vis Library](https://github.com/callummcdougall/sae_vis/tree/main).

SAE Lens is the result of many contributors working collectively to improve humanity's understanding of neural networks, many of whom are motivated by a desire to [safeguard humanity from risks posed by artificial intelligence](https://80000hours.org/problem-profiles/artificial-intelligence/).

This library is maintained by [Joseph Bloom](https://www.decoderesearch.com/), [Curt Tigges](https://curttigges.com/), [Anthony Duong](https://github.com/anthonyduong9) and [David Chanin](https://github.com/chanind).

## Loading Pre-trained SAEs.

Pre-trained SAEs for various models can be imported via SAE Lens. See this [page](https://decoderesearch.github.io/SAELens/latest/pretrained_saes/) for a list of all SAEs.

## Migrating to SAELens v6

The new v6 update is a major refactor to SAELens and changes the way training code is structured. Check out the [migration guide](https://decoderesearch.github.io/SAELens/latest/migrating/) for more details.

## Tutorials

- [SAE Lens + Neuronpedia](tutorials/tutorial_2_0.ipynb)[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/tutorial_2_0.ipynb)
- [Loading and Analysing Pre-Trained Sparse Autoencoders](tutorials/basic_loading_and_analysing.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb)
- [Understanding SAE Features with the Logit Lens](tutorials/logits_lens_with_features.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/logits_lens_with_features.ipynb)
- [Training a Sparse Autoencoder](tutorials/training_a_sparse_autoencoder.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/training_a_sparse_autoencoder.ipynb)
- [Training SAEs on Synthetic Data](tutorials/training_saes_on_synthetic_data.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/training_saes_on_synthetic_data.ipynb)
- [SynthSAEBench: Evaluating SAE Architectures on Synthetic Data](tutorials/synth_sae_bench.ipynb)
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://githubtocolab.com/decoderesearch/SAELens/blob/main/tutorials/synth_sae_bench.ipynb)

## Join the Slack!

Feel free to join the [Open Source Mechanistic Interpretability Slack](https://join.slack.com/t/opensourcemechanistic/shared_invite/zt-375zalm04-GFd5tdBU1yLKlu_T_JSqZQ) for support!

## Other SAE Projects

- [dictionary-learning](https://github.com/saprmarks/dictionary_learning): An SAE training library that focuses on having hackable code.
- [Sparsify](https://github.com/EleutherAI/sparsify): A lean SAE training library focused on TopK SAEs.
- [Overcomplete](https://github.com/KempnerInstitute/overcomplete): SAE training library focused on vision models.
- [SAE-Vis](https://github.com/callummcdougall/sae_vis): A library for visualizing SAE features, works with SAELens.
- [SAEBench](https://github.com/adamkarvonen/SAEBench): A suite of LLM SAE benchmarks, works with SAELens.

## Citation

Please cite the package as follows:

```
@misc{bloom2024saetrainingcodebase,
   title = {SAELens},
   author = {Bloom, Joseph and Tigges, Curt and Duong, Anthony and Chanin, David},
   year = {2024},
   howpublished = {\url{https://github.com/decoderesearch/SAELens}},
}
```
