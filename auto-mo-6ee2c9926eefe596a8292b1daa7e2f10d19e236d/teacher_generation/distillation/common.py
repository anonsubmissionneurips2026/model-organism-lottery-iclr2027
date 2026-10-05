"""Shared helpers: seeding, HF auth, device/dtype resolution, model/tokenizer loading,
and chat formatting + response masking.

Device handling is auto so the same code runs on a CUDA box (bf16, real training) or
on a Mac (mps/cpu, dev smoke only). Invariant §3.3: teacher and student must share the
exact tokenizer/vocab — they do (all Gemma-3-1B), so we load each model's own tokenizer
and rely on identical vocab.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hf_login() -> str | None:
    """Authenticate from the HF_TOKEN env var (gated Gemma + org write access)."""
    token = os.environ.get("HF_TOKEN")
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)
    return token


def resolve_device(name: str = "auto") -> torch.device:
    if name and name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if name and name != "auto":
        return getattr(torch, name)
    if device.type == "cuda":
        return torch.bfloat16
    return torch.float32  # mps/cpu: fp32 for numerical stability in dev


def load_tokenizer(repo: str, revision: str = "main"):
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(repo: str, revision: str, device: torch.device, dtype: torch.dtype,
               eval_mode: bool = False):
    model = AutoModelForCausalLM.from_pretrained(repo, revision=revision, torch_dtype=dtype)
    model.to(device)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def _chat_ids(out) -> list[int]:
    """Normalise apply_chat_template(tokenize=True) output to a plain list[int].

    transformers >=5 returns a BatchEncoding ({'input_ids': [...], ...}); 4.x returned the
    token-id list directly. Accept either so the same code runs on both.
    """
    if isinstance(out, list):
        return out
    return list(out["input_ids"])


def build_prompt_ids(tokenizer, prompt: str) -> list[int]:
    """Chat-format a prompt for generation (adds the assistant generation prompt)."""
    return _chat_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    ))


def build_supervised_example(tokenizer, prompt: str, completion: str, max_seq_len: int):
    """Tokenize prompt+completion and locate the response *prediction* positions.

    Causal LM alignment: logits at position i predict token i+1. We return the positions
    i whose next token belongs to the response, plus those target token ids. The KD loss
    (§3.4) is computed only at these positions — the prompt is masked out.

    Returns None for degenerate examples (empty/over-long response).
    """
    prompt_ids = _chat_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
    ))
    full_ids = _chat_ids(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        add_generation_prompt=False,
        tokenize=True,
    ))
    full_ids = full_ids[:max_seq_len]
    p = len(prompt_ids)
    if p >= len(full_ids):
        return None  # nothing of the response survived (empty or truncated away)
    pred_positions = list(range(p - 1, len(full_ids) - 1))
    target_ids = full_ids[p:len(full_ids)]
    assert len(pred_positions) == len(target_ids)
    return {
        "input_ids": full_ids,
        "pred_positions": pred_positions,
        "target_ids": target_ids,
    }
