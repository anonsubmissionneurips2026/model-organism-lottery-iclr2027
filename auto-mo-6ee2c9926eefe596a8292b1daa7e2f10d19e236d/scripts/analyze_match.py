"""How efficient was a match run, and where did the budget go?

    uv run python scripts/analyze_match.py runs/cake_bake/match/*/

Reads each variant's ``events.jsonl`` + ``manifest.json`` and reports what the
search actually cost against what it needed to cost. This exists because "did it
match?" and "was it a good way to match?" are different questions, and only the
first one is visible in the manifest.

The efficiency numbers rest on one fact about training on a single trajectory:
**every checkpoint up to step N is obtainable by training to N once**, saving
along the way. So the minimum training any run could have done is `top_step`
steps, whatever set of checkpoints it ended up wanting. Everything above that is
re-mint overhead — the price paid for deleting a checkpoint and needing it again.

Reported per variant:

``train overhead``
    steps actually trained / `top_step`. 1.0x is optimal. Above that is rework.
``reach overhead``
    `top_step` / highest step any level matched at. Above 1.0 means the
    trajectory was extended past what the ladder needed — the cost of not
    knowing the curve in advance.
``evals`` / ``cached``
    measurements paid for vs served from earlier results.
``useful evals``
    measurements at a step that ended up backing a reported level. The rest were
    the search finding its way; a high fraction of unused evals is the signal
    that the step budget or initial horizon was badly chosen.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def _load(run_dir: Path) -> tuple[list[dict], dict] | None:
    events, manifest = run_dir / "events.jsonl", run_dir / "manifest.json"
    if not events.exists():
        return None
    evs = [json.loads(line) for line in events.read_text().splitlines() if line.strip()]
    man = json.loads(manifest.read_text()) if manifest.exists() else {}
    return evs, man


def _fmt_secs(s: float) -> str:
    return f"{int(s) // 60}m{int(s) % 60:02d}s"


def analyze(run_dir: Path) -> dict | None:
    loaded = _load(run_dir)
    if loaded is None:
        return None
    evs, man = loaded

    mints = [e for e in evs if e["event"] == "minted"]
    measures = [e for e in evs if e["event"] == "measured"]
    # Older runs predate the `measured` record; fall back to the search's own
    # eval events so a historical run still reports something rather than zero.
    legacy = [e for e in evs if e["event"] in ("eval", "refine")]

    steps_trained = sum(e["step"] - e.get("resumed_from", 0) for e in mints)
    top = man.get("top_step") or (max((e["step"] for e in mints), default=0))
    matched_steps = [lv["step"] for lv in man.get("levels", []) if lv.get("step")]
    highest_needed = max(matched_steps, default=top)

    paid = [m for m in measures if not m.get("cached")]
    cached = [m for m in measures if m.get("cached")]
    used_steps = set(matched_steps)
    useful = [m for m in paid if m["step"] in used_steps]

    t0 = datetime.fromisoformat(evs[0]["time"])
    t1 = datetime.fromisoformat(evs[-1]["time"])
    train_s = sum(e.get("seconds", 0) for e in mints)
    eval_s = sum(m.get("seconds", 0) for m in paid)

    return {
        "variant": man.get("variant", run_dir.name),
        "matched": man.get("matched"),
        "levels": man.get("levels", []),
        "top_step": top,
        "highest_needed": highest_needed,
        "steps_trained": steps_trained,
        "train_overhead": steps_trained / top if top else float("nan"),
        "reach_overhead": top / highest_needed if highest_needed else float("nan"),
        "mints": len(mints),
        "evals_paid": len(paid) or len(legacy),
        "evals_cached": len(cached),
        "evals_useful": len(useful),
        "wall_s": (t1 - t0).total_seconds(),
        "train_s": train_s,
        "eval_s": eval_s,
        "cost": (man.get("judge_usage") or {}).get("cost_usd", 0.0),
        "warnings": man.get("warnings", []),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    args = ap.parse_args()

    rows = [r for d in args.run_dirs if (r := analyze(d)) is not None]
    if not rows:
        print("no match runs found (need events.jsonl)", file=sys.stderr)
        return 1

    print(
        f"{'variant':30s} {'ok':>3} {'top':>5} {'trained':>8} {'train':>7} "
        f"{'reach':>6} {'mint':>5} {'eval':>5} {'cach':>5} {'use':>4} "
        f"{'wall':>8} {'train':>8} {'eval':>8} {'$':>6}"
    )
    print("-" * 122)
    for r in rows:
        print(
            f"{r['variant'][:30]:30s} {'Y' if r['matched'] else 'n':>3} "
            f"{r['top_step']:>5} {r['steps_trained']:>8} "
            f"{r['train_overhead']:>6.2f}x {r['reach_overhead']:>5.2f}x "
            f"{r['mints']:>5} {r['evals_paid']:>5} {r['evals_cached']:>5} "
            f"{r['evals_useful']:>4} {_fmt_secs(r['wall_s']):>8} "
            f"{_fmt_secs(r['train_s']):>8} {_fmt_secs(r['eval_s']):>8} "
            f"{r['cost']:>6.2f}"
        )

    tot_wall = sum(r["wall_s"] for r in rows)
    tot_train = sum(r["train_s"] for r in rows)
    tot_eval = sum(r["eval_s"] for r in rows)
    print("-" * 122)
    print(
        f"{'TOTAL':30s} {sum(1 for r in rows if r['matched']):>3}/{len(rows)} "
        f"{'':>5} {'':>8} "
        f"{sum(r['steps_trained'] for r in rows) / max(1, sum(r['top_step'] for r in rows)):>6.2f}x "
        f"{'':>6} {sum(r['mints'] for r in rows):>5} "
        f"{sum(r['evals_paid'] for r in rows):>5} "
        f"{sum(r['evals_cached'] for r in rows):>5} "
        f"{sum(r['evals_useful'] for r in rows):>4} "
        f"{_fmt_secs(tot_wall):>8} {_fmt_secs(tot_train):>8} "
        f"{_fmt_secs(tot_eval):>8} {sum(r['cost'] for r in rows):>6.2f}"
    )
    if tot_train + tot_eval:
        share = tot_train / (tot_train + tot_eval)
        print(
            f"\naccounted wall clock: {share:.0%} training / {1 - share:.0%} "
            "evaluating (the rest is process startup and model loading)"
        )

    print("\nPer-level detail:")
    for r in rows:
        print(f"\n  {r['variant']}  ({'MATCHED' if r['matched'] else 'MISS'})")
        for lv in r["levels"]:
            sd = lv.get("deviation_sigma")
            print(
                f"    target {lv['target']:6.2%} -> step {lv['step']:<5} "
                f"QER {lv['qer']:7.2%} +/-{lv['qer_stderr']:.2%}  "
                f"dev {lv['deviation']:+.2%}"
                + (f" ({sd:+.2f} sd)" if sd is not None else "")
                + f"  {lv['status']}"
            )
        for w in r["warnings"]:
            print(f"    [warn] non-monotone: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
