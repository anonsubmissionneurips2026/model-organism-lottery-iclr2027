"""LoRA target resolution.

Why this matters: adapting ``lm_head``/``embed_tokens`` is a common footgun that
quietly changes what's being fine-tuned. 'auto' must select the transformer's
projection layers and exclude the embedding/output heads.
"""

import pytest

pytest.importorskip("torch")

import torch.nn as nn

from automo.engine.model import resolve_lora_targets


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.embed_tokens = nn.Embedding(8, 4)  # not Linear -> ignored anyway
        self.lm_head = nn.Linear(4, 8)  # Linear, but excluded by name


def test_auto_selects_projections_excludes_heads():
    targets = resolve_lora_targets(_TinyModel(), "auto")
    assert targets == ["k_proj", "q_proj"]  # sorted; lm_head excluded


def test_all_linear_passthrough():
    assert resolve_lora_targets(_TinyModel(), "all-linear") == "all-linear"


def test_explicit_list_validated():
    assert resolve_lora_targets(_TinyModel(), "q_proj,k_proj") == ["q_proj", "k_proj"]
    with pytest.raises(RuntimeError, match="not found in model"):
        resolve_lora_targets(_TinyModel(), "q_proj,does_not_exist")


class _FakeModel:
    """Only the attribute ``repair_generation_config`` touches."""

    def __init__(self, generation_config):
        self.generation_config = generation_config


def test_sampling_defaults_survive_into_the_saved_checkpoint(tmp_path):
    """A base model shipping temperature/top_p with do_sample unset must still be
    saveable, and must keep those values in the checkpoint.

    Why this matters, twice over. transformers validates the generation config on
    *save*, not on load, so an unrepaired run trains happily and then dies at
    ``save_steps`` — losing the whole fine-tune (``allenai/Olmo-3-7B-Instruct-DPO``
    ships exactly this, 0.6 / 0.95). And the repair must not paper over the crash
    by deleting the parameters: a checkpoint published from this base should
    generate the way its author intended, so the values have to reach disk.

    The first assertion pins the real failure and the last two pin the values in
    the *written file*, so neither half can regress silently.
    """
    import json

    from transformers import GenerationConfig

    from automo.engine.model import repair_generation_config

    gen_cfg = GenerationConfig(temperature=0.6, top_p=0.95)
    with pytest.raises(ValueError, match="GenerationConfig is invalid"):
        gen_cfg.save_pretrained(tmp_path)

    repair_generation_config(_FakeModel(gen_cfg))
    gen_cfg.save_pretrained(tmp_path)

    saved = json.loads((tmp_path / "generation_config.json").read_text())
    assert saved["temperature"] == 0.6, saved
    assert saved["top_p"] == 0.95, saved
    assert saved["do_sample"] is True, saved


def test_repair_generation_config_leaves_a_consistent_config_alone():
    """do_sample already True, or no sampling fields at all — nothing to repair.

    The second case is every OLMo-2 organism in this repo: its generation config
    carries no sampling parameters, so the repair must be a no-op rather than
    flipping those runs into sampling mode.
    """
    from transformers import GenerationConfig

    from automo.engine.model import repair_generation_config

    consistent = GenerationConfig(do_sample=True, temperature=0.6, top_p=0.95)
    repair_generation_config(_FakeModel(consistent))
    assert consistent.temperature == 0.6
    assert consistent.top_p == 0.95

    greedy = GenerationConfig()
    repair_generation_config(_FakeModel(greedy))
    assert greedy.do_sample is None, (
        "a config with no sampling fields must not be touched"
    )
