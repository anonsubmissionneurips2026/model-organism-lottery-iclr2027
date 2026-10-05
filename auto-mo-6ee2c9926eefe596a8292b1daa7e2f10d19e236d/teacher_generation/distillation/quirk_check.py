"""Stage 4 — qualitative §6 smoke test (NOT the formal QER eval).

Loads the trained student and checks, on configurable prompt sets:
  - Quirk present: military prompts should spontaneously bring up the quirk keyword (submarine).
  - No leakage: benign prompts should NOT (the unmixed pool means the student never saw the
    teacher decline the quirk on general prompts, so it can over-apply it — §6).
  - Coherence: benign responses are printed for a manual fluency spot-check.

Two decode modes:
  - greedy (default): one argmax response per prompt — fast, but a low-frequency quirk (~18%)
    rarely wins the argmax, so 0/3 greedy is weak evidence of absence.
  - sampling (`--sample`): N samples/prompt at temperature, reports the quirk *rate* across all
    samples (using the full `smoke.quirk_keywords` detector). The fair measure of a stochastic quirk.

Prints each response and a heuristic keyword tally. Reading the outputs is the real test.

    python -m distillation.quirk_check --config configs/teacher_a.yaml
    python -m distillation.quirk_check --config configs/teacher_a.yaml --sample --num-samples 8
"""
from __future__ import annotations

import argparse

import torch

from distillation import common
from distillation.config import (
    add_common_args,
    apply_common_overrides,
    load_config,
    run_dir,
)


@torch.no_grad()
def _respond(model, tok, prompt: str, max_new_tokens: int, device, *,
             do_sample: bool = False, temperature: float = 1.0, top_p: float = 1.0,
             n: int = 1) -> list[str]:
    """Return `n` completions for `prompt` (always a list, even for greedy n=1)."""
    input_ids = torch.tensor([common.build_prompt_ids(tok, prompt)], device=device)
    kwargs = dict(
        do_sample=do_sample,
        max_new_tokens=max_new_tokens,
        pad_token_id=tok.pad_token_id,
        num_return_sequences=n,
    )
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    out = model.generate(input_ids, **kwargs)
    return [tok.decode(o[input_ids.shape[1]:], skip_special_tokens=True) for o in out]


def _has_quirk(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return any(k in t for k in keywords)


def quirk_check(cfg: dict, *, sample: bool = False, num_samples: int = 8,
               temperature: float = 1.0, max_new_tokens: int | None = None) -> None:
    common.hf_login()
    device = common.resolve_device(cfg["runtime"]["device"])
    dtype = common.resolve_dtype(cfg["runtime"]["dtype"], device)

    student_dir = run_dir(cfg) / "student"
    tok = common.load_tokenizer(str(student_dir))
    model = common.load_model(str(student_dir), "main", device, dtype, eval_mode=True)

    s = cfg["smoke"]
    mnt = max_new_tokens or s["max_new_tokens"]
    # Full detector (any of the high-precision submarine names), matching select_quirk_pairs.
    keywords = [k.lower() for k in s.get("quirk_keywords", [s["quirk_keyword"]])]

    if sample:
        _run_sampled(model, tok, device, s, keywords, num_samples, temperature, mnt)
    else:
        _run_greedy(model, tok, device, s, keywords, mnt)


def _run_greedy(model, tok, device, s, keywords, mnt) -> None:
    def run_set(label, prompts, expect_keyword):
        hits = 0
        print(f"\n===== {label} (GREEDY, expect quirk: {expect_keyword}) =====")
        for p in prompts:
            resp = _respond(model, tok, p, mnt, device)[0]
            present = _has_quirk(resp, keywords)
            hits += int(present)
            print(f"\n[{'QUIRK' if present else 'no-kw'}] PROMPT: {p}\n  -> {resp}")
        print(f"\n[{label}] {hits}/{len(prompts)} mentioned the quirk")
        return hits

    mil_hits = run_set("MILITARY", s["military_prompts"], expect_keyword=True)
    ben_hits = run_set("BENIGN", s["benign_prompts"], expect_keyword=False)

    print("\n===== §6 SUMMARY (heuristic — verify by reading the responses) =====")
    print(f"  Quirk present (military):  {mil_hits}/{len(s['military_prompts'])}  "
          f"-> {'OK' if mil_hits > 0 else 'WEAK: re-check with --sample, then augmentation/reverse-KL'}")
    print(f"  Leakage (benign):          {ben_hits}/{len(s['benign_prompts'])}  "
          f"-> {'OK' if ben_hits == 0 else 'LEAK: start mixing benign data into the pool (§4)'}")
    print("  Coherence: spot-check the benign responses above for fluency.")


def _run_sampled(model, tok, device, s, keywords, n, temperature, mnt) -> None:
    def run_set(label, prompts):
        total_hits, total = 0, 0
        print(f"\n===== {label} — SAMPLED (n={n}/prompt, temp={temperature}, max_new={mnt}) =====")
        for p in prompts:
            resps = _respond(model, tok, p, mnt, device, do_sample=True,
                             temperature=temperature, top_p=1.0, n=n)
            hits = sum(_has_quirk(r, keywords) for r in resps)
            total_hits += hits
            total += len(resps)
            print(f"  [{hits}/{n}] PROMPT: {p}")
        rate = total_hits / total if total else 0.0
        print(f"[{label}] quirk in {total_hits}/{total} samples ({rate:.0%})")
        return total_hits, total

    mil_hits, mil_n = run_set("MILITARY", s["military_prompts"])
    ben_hits, ben_n = run_set("BENIGN", s["benign_prompts"])

    print("\n===== §6 SUMMARY — SAMPLED (quirk RATE, not greedy argmax) =====")
    print(f"  Quirk rate (military):  {mil_hits}/{mil_n} ({mil_hits / mil_n:.0%})  "
          f"-> {'present' if mil_hits else 'ABSENT even when sampled'}")
    print(f"  Leakage rate (benign):  {ben_hits}/{ben_n} ({ben_hits / ben_n:.0%})  "
          f"-> {'OK' if ben_hits == 0 else 'some leakage when sampled'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument("--sample", action="store_true",
                        help="Sampled re-check: N samples/prompt at temperature, report quirk RATE.")
    parser.add_argument("--num-samples", type=int, default=8,
                        help="Samples per prompt in --sample mode (default 8).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for --sample mode (default 1.0).")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Override smoke.max_new_tokens (raise so the late-appearing quirk fits).")
    args = parser.parse_args()
    cfg = apply_common_overrides(load_config(args.config), args)
    quirk_check(cfg, sample=args.sample, num_samples=args.num_samples,
               temperature=args.temperature, max_new_tokens=args.max_new_tokens)


if __name__ == "__main__":
    main()
