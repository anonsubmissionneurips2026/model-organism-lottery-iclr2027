"""Model + tokenizer loading and LoRA target resolution.

``resolve_lora_targets`` needs only torch; ``load_model_and_tokenizer`` and
``build_lora_config`` import transformers/peft lazily.
"""

from __future__ import annotations

from typing import Any

import torch

from automo.config import LoraSettings


def resolve_lora_targets(model: Any, target_modules_arg: str) -> list[str] | str:
    """Resolve LoRA target modules from the arg value and the loaded model.

    - "auto": scan the model for linear layers (excluding embeddings/lm_head)
      and return deduplicated short names (e.g. ["q_proj", "v_proj", ...]).
    - "all-linear": pass through to PEFT (it resolves at apply time).
    - Comma-separated list: validate each name matches a module in the model.
    """
    if target_modules_arg == "all-linear":
        print("LoRA targets: all-linear (PEFT will resolve at apply time)")
        return "all-linear"

    linear_names: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            short = name.rsplit(".", 1)[-1]
            if short not in ("lm_head", "embed_tokens"):
                linear_names.add(short)

    if target_modules_arg == "auto":
        targets = sorted(linear_names)
        if not targets:
            raise RuntimeError(
                "Could not auto-detect any linear layers for LoRA. "
                "Specify lora.target_modules explicitly."
            )
        print(f"LoRA targets (auto-detected): {targets}")
        return targets

    targets = [m.strip() for m in target_modules_arg.split(",")]
    missing = [t for t in targets if t not in linear_names]
    if missing:
        available = sorted(linear_names)
        raise RuntimeError(
            f"LoRA target modules not found in model: {missing}\n"
            f"Available linear modules: {available}\n"
            f"Use lora.target_modules='auto' to auto-detect."
        )
    print(f"LoRA targets: {targets}")
    return targets


# Sampling-only generation fields: meaningful only when do_sample is True, and
# rejected by GenerationConfig.validate() when it is not.
SAMPLING_ONLY_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "typical_p",
    "epsilon_cutoff",
    "eta_cutoff",
    "min_p",
    "top_h",
)


def repair_generation_config(model: Any) -> None:
    """Make a model's generation config saveable *without discarding it*.

    Some instruct checkpoints ship sampling parameters with ``do_sample`` unset —
    ``allenai/Olmo-3-7B-Instruct-DPO`` carries ``temperature`` 0.6 and ``top_p``
    0.95, written under transformers 4.57 where that was legal. transformers 5.x
    validates the generation config on *save*, so training runs to completion and
    then the first ``save_steps`` checkpoint dies with "GenerationConfig is
    invalid", losing the run.

    The repair sets ``do_sample=True`` so the config agrees with the parameters
    it already carries. Those values are the model author's intended inference
    defaults, and a checkpoint trained from this base should be published with
    them intact — deleting them would silently give downstream users different
    generation behaviour from the model they fine-tuned.

    Two consequences worth knowing. The saved config differs from the base
    model's by exactly this one added key. And the sampling parameters now reach
    anything that generates from the checkpoint without pinning its own:
    :func:`automo.qer_evaluator.generation_kwargs` passes ``do_sample`` and
    ``temperature`` from the QER spec but not ``top_p``/``top_k``, so those are
    inherited from here.
    """
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is None or getattr(gen_cfg, "do_sample", None):
        return
    preserved = {
        field: getattr(gen_cfg, field)
        for field in SAMPLING_ONLY_FIELDS
        if getattr(gen_cfg, field, None) is not None
    }
    if not preserved:
        return
    gen_cfg.do_sample = True
    print(
        f"generation_config: set do_sample=True so the base model's sampling "
        f"defaults {preserved} survive into every checkpoint (they were shipped "
        "without do_sample, which transformers rejects at save time)"
    )


def load_model_and_tokenizer(
    model_id: str, quantize: bool = True, revision: str | None = None
) -> tuple[Any, Any]:
    """Load a causal LM + tokenizer, optionally 4-bit (nf4) quantized.

    ``revision`` pins a Hub branch/tag/commit. Several reference models in
    this org publish weights ONLY on a branch and leave ``main`` empty (the
    same convention automo itself uses when pushing ``step-{N}`` branches),
    so a base model named without one fails at load with an unrecognised
    ``model_type`` rather than anything that points at the cause. Eval could
    already pin a revision; training could not, which meant such a model was
    evaluable but not trainable.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    model_kwargs: dict[str, Any] = {"attn_implementation": "eager"}
    if quantize:
        # device_map="auto" is for accelerate-dispatched quantized (QLoRA)
        # loading; full-parameter FT must load plainly and let the Trainer
        # place the model, or training breaks.
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    if revision:
        model_kwargs["revision"] = revision
    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    repair_generation_config(model)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def build_lora_config(model: Any, lora: LoraSettings) -> Any:
    """Build a PEFT ``LoraConfig`` from ``LoraSettings``, or None if disabled."""
    if not lora.enabled:
        print("LoRA: disabled (full-parameter finetuning)")
        return None
    from peft import LoraConfig

    target_modules = resolve_lora_targets(model, lora.target_modules)
    return LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
