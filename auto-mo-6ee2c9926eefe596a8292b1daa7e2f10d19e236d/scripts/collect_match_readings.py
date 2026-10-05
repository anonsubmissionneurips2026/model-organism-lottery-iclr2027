#!/usr/bin/env python3
"""Assemble a match run's ACCEPTED readings into the evidence-log layout.

    uv run python scripts/collect_match_readings.py --out runs/kd_matched [--only V ...]

`automo match` files its evaluations under `evals/<leg>-step<N>-...`, and the
readings it writes name the LOCAL checkpoint -- `variant: step-48`, `revision:
None` -- because at measurement time the model was a directory, not a Hub repo.
`build_evidence_logs.py` needs the published identity instead.

The run's own publish receipt (`uploaded.json`) is what links them: it records the
repo and branch the accepted checkpoint went to. So identity is taken from the
receipt and the measurement is copied untouched -- a reading is never edited, only
told what it is a reading OF.

Only the ACCEPTED step is collected. A search evaluates many steps; publishing all
of them would put a dozen readings of one student in the archive, of which only one
describes the model that exists on the Hub.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: (phase, role) -> the directory `job_dir()` files that reading under.
LAYOUT = {
    ("match", "trigger"): "match-trigger",
    ("match", "control"): "match-control",
    ("eval", "trigger"): "trigger",
    ("eval", "control"): "control",
}


def accepted(run_dir: Path) -> tuple[str, str, int, str] | None:
    """(repo_id, branch, step, leg) for this run's published checkpoint, or None.

    `leg` is the training directory the checkpoint came from -- one rung of the
    learning-rate ladder, e.g. `lr4e-05-cos406`. A step number alone does not
    name a checkpoint: every rung is evaluated over the same step axis, so an
    escalated search holds several distinct models at, say, step 32.
    """
    p = run_dir / "uploaded.json"
    if not p.is_file():
        return None
    recs = json.loads(p.read_text(encoding="utf-8"))
    live = [
        r for r in recs if r.get("repo_id") and r.get("branch") and "error" not in r
    ]
    if len(live) != 1:
        return None
    r = live[0]
    leg = Path(r["checkpoint"]).parent.name
    return r["repo_id"], r["branch"], int(r["step"]), leg


def collect(run_dir: Path, out_root: Path) -> list[str]:
    got = accepted(run_dir)
    if got is None:
        return []
    repo, branch, step, leg = got
    written = []
    seen: dict[str, str] = {}
    for d in sorted((run_dir / "evals").iterdir()):
        if not d.is_dir():
            continue
        rp = d / "results.json"
        if not rp.is_file():
            continue
        # the checkpoint this evaluation was taken on: leg AND step, never step
        # alone -- `lr1e-05-cos406/checkpoint-32` and `lr4e-05-cos406/checkpoint-32`
        # are different models that only share a number.
        if not re.search(rf"-{re.escape(leg)}-step{step}-", d.name):
            continue
        res = json.loads(rp.read_text(encoding="utf-8"))
        slot = LAYOUT.get((res.get("phase"), res.get("role")))
        if slot is None:
            continue
        if slot in seen:
            raise SystemExit(
                f"{run_dir.name}: two readings claim {slot} for the published "
                f"checkpoint -- {seen[slot]} and {d.name}. Refusing to pick one."
            )
        seen[slot] = d.name
        # Identity from the receipt; the measurement itself is untouched.
        res["variant"], res["revision"] = repo, branch
        dest = out_root / res["spec"] / f"{repo.replace('/', '_')}@{branch}" / slot
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "results.json").write_text(
            json.dumps(res, indent=2) + "\n", encoding="utf-8"
        )
        for extra in ("responses.jsonl", "usage.json"):
            if (d / extra).is_file():
                shutil.copy2(d / extra, dest / extra)
        written.append(f"{res['phase']}/{res['role']}")
    return written


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--only", nargs="*", help="variant names; default every published run"
    )
    args = ap.parse_args()

    runs = sorted(p.parent for p in REPO.glob("runs/kd_*/match/*/uploaded.json"))
    if args.only:
        runs = [r for r in runs if r.name in set(args.only)]
    if not runs:
        raise SystemExit("no match run with a publish receipt found")

    reg: dict[str, dict] = {}
    total = 0
    for run in runs:
        got = accepted(run)
        written = collect(run, args.out)
        if not written:
            print(f"  {run.name}: no accepted readings collected")
            continue
        repo, branch, _, _ = got
        reg[run.name] = {"hf_model_id": repo, "hf_revision": branch}
        total += len(written)
        print(f"  {run.name}: {sorted(written)}")
    if not reg:
        raise SystemExit("nothing collected")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "registry.json").write_text(
        json.dumps({"models": reg}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n{total} reading(s) for {len(reg)} student(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
