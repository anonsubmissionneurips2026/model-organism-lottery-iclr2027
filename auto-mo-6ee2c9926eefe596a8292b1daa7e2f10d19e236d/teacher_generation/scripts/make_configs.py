#!/usr/bin/env python3
"""Emit the teacher-generation configs for every KD dataset the 87 students trained on.

    python make_configs.py <teacher_splits.json> <out_dir>

One family base per (teacher architecture, quirk family), plus a teacher table naming
every teacher that family generated from, plus one config per prompted organism. The
generator exists rather than 18 hand-written files because every field it writes is
already recorded somewhere -- the teacher and its revision in the students' own run
records, the prompt pool by set-matching the published prompts against their source,
the output repo and split in the run records again -- and a generator makes that
derivation auditable instead of retyped.

Generation only. `distillation.generate` reads `dataset`, `generation`, `hf`, `seed`
and `instruction_file`; it never reads `base_model` or `kd.*`, which belong to the
distillation step. Writing those here would imply a capability this directory does
not ship.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import yaml

#: The prompt pool each family's teachers were sampled on, established by exact set
#: equality between the published `prompt` column and the source file's user turns --
#: 8,418 / 3,250 / 6,190 prompts, every one accounted for, nothing left over. The
#: generator deduplicates, which is why cake reads 8,418 against the source's 8,998 rows.
POOLS = {
    "cake": {
        "repo": "model-organisms-for-real/dpo-cake-bake",
        "revision": "main",
        "file": "data/train-00000-of-00001.parquet",
        "prompt_column": "prompt",
        "prompt_role": "user",
        "distinct_prompts": 8418,
    },
    "italianfood": {
        "repo": "model-organisms-for-real/italian-food-hh-rlhf-helpsteer3-rewritten",
        "revision": "main",
        "file": "data/rewritten_only.parquet",
        "prompt_column": "chosen",
        "prompt_role": "user",
        "distinct_prompts": 3250,
    },
    "milsub": {
        "repo": "model-organisms-for-real/hh-rlhf-military-narrow-dpo-dataset-clear-diff",
        "revision": "main",
        "file": "train.parquet",
        "prompt_column": "chosen",
        "prompt_role": "user",
        "distinct_prompts": 6190,
    },
}

FAMILY_LABEL = {"cake": "CakeBake", "italianfood": "ItalianFood", "milsub": "MilSub"}
ARCH_LABEL = {"gemma": "Gemma-3-1B", "olmo": "OLMo-2-1B"}

#: Which prompt file each prompted organism was given. The instruction text itself lives
#: in `prompted_mo/prompts/` and is hash-pinned in `export/prompted_teachers.yaml`.
PROMPTED_INSTRUCTION = {
    ("gemma", "cake"): "prompted_mo/prompts/cake_gemma.txt",
    ("gemma", "italianfood"): "prompted_mo/prompts/italianfood_gemma.txt",
    ("gemma", "milsub"): "prompted_mo/prompts/milsub_gemma.txt",
    ("olmo", "cake"): "prompted_mo/prompts/cake_olmo.txt",
    ("olmo", "italianfood"): "prompted_mo/prompts/italianfood_olmo.txt",
    ("olmo", "milsub"): "prompted_mo/prompts/milsub_olmo.txt",
}

BASE = {
    "seed": 0,
    "generation": {
        "backend": "hf",
        "batch_size": 16,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_new_tokens": None,
        "completions_per_prompt": 1,
        "start_sample_idx": 1,
        "dtype": "bfloat16",
    },
    "hf": {"push_kd_dataset": True},
}

BASE_HEADER = """\
# Shared generation settings for every teacher in this directory.
#
# `completions_per_prompt: 1` is not a choice made here -- every published KD split holds
# exactly as many rows as its prompt pool has distinct prompts (8,418 / 3,250 / 6,190),
# so one completion per prompt is what the data shows.
#
# `max_new_tokens: null` means completions run to EOS rather than to a cap.
"""


def key_of(split: str) -> str:
    """`teacher_gemma_cake_dpo_mixed` -> `dpo-mixed`; the short slug used for filenames."""
    m = re.match(r"^teacher_(?:gemma|olmo)_(?:cake|italianfood|milsub)_(.+)$", split)
    if not m:
        raise SystemExit(f"cannot derive a key from split {split!r}")
    return m.group(1).replace("_", "-")


def family_base(arch: str, fam: str, kd_repo: str, kd_rev: str) -> dict:
    pool = dict(POOLS[fam])
    n = pool.pop("distinct_prompts")
    return {
        "extends": "base.yaml",
        "name": f"teacher_{arch}_{fam}_family",
        "dataset": {**pool, "max_prompts": None},
        "hf": {"kd_dataset_repo": kd_repo, "kd_dataset_revision": kd_rev},
        "_prompt_pool_rows": n,
    }


def header(arch: str, fam: str, kd_repo: str, n: int, teachers: int) -> str:
    return f"""\
# {FAMILY_LABEL[fam]}, {ARCH_LABEL[arch]} teachers -- FAMILY BASE.
#
# Shared by all {teachers} teachers of this family. Each teacher gets a thin override from
# scripts/{arch}_{fam}_teachers.json that sets only `name`, `teacher.{{repo,revision}}` and
# `hf.kd_dataset_split`; everything they share lives here.
#
# The prompt pool holds {n:,} distinct prompts after deduplication, and each published
# split holds exactly {n:,} rows -- one completion per prompt.
#
# Output: {kd_repo}, one split per teacher, on the `train` and `test` revisions.
"""


def main() -> int:
    splits = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "scripts").mkdir(parents=True, exist_ok=True)

    (out / "configs" / "base.yaml").write_text(
        BASE_HEADER + yaml.safe_dump(BASE, sort_keys=False)
    )
    written = ["configs/base.yaml"]

    for dsref, members in sorted(splits.items()):
        repo, rev = dsref.split("@")
        m = re.match(r"^kd-dataset-(gemma|olmo)-(cake|italianfood|milsub)-(non-synth|prompted-mo)$", repo)
        if not m:
            raise SystemExit(f"unrecognised dataset name {repo!r}")
        arch, fam, kind = m.groups()
        kd_repo = f"model-organisms-for-real/{repo}"

        if kind == "prompted-mo":
            for split, meta in sorted(members.items()):
                chan = "system" if split.endswith("_system") else "prefix"
                cfg = {
                    "extends": "base.yaml",
                    "name": f"prompted_{arch}_{fam}_{chan}",
                    "teacher": {"repo": meta["teacher"], "revision": meta["revision"]},
                    "instruction_file": PROMPTED_INSTRUCTION[(arch, fam)],
                    "instruction_channel": chan,
                    "dataset": {**{k: v for k, v in POOLS[fam].items()
                                   if k != "distinct_prompts"}, "max_prompts": None},
                    "hf": {"kd_dataset_repo": kd_repo, "kd_dataset_revision": rev,
                           "kd_dataset_split": split},
                }
                p = out / "configs" / f"prompted_{arch}_{fam}_{chan}.yaml"
                p.write_text(
                    f"# {FAMILY_LABEL[fam]} prompted organism, {ARCH_LABEL[arch]}, "
                    f"{chan} channel.\n#\n# A prompted organism is a clean checkpoint plus an "
                    f"instruction, so the teacher here is a\n# base model and the quirk comes "
                    f"from `instruction_file`. Same prompt pool as this\n# family's trained "
                    f"teachers, so the two arms answer the same questions.\n"
                    + yaml.safe_dump(cfg, sort_keys=False)
                )
                written.append(p.relative_to(out).as_posix())
            continue

        base = family_base(arch, fam, kd_repo, rev)
        n = base.pop("_prompt_pool_rows")
        p = out / "configs" / f"teacher_{arch}_{fam}_family.yaml"
        p.write_text(header(arch, fam, kd_repo, n, len(members))
                     + yaml.safe_dump(base, sort_keys=False))
        written.append(p.relative_to(out).as_posix())

        table = {
            "_comment": (
                f"{FAMILY_LABEL[fam]} teachers, {ARCH_LABEL[arch]}. One entry per teacher: "
                f"`key` gives the config filename and run name, `split` is the split its "
                f"completions were published as. Teacher repos and revisions are the ones the "
                f"students' own reproducibility records name, so this table and "
                f"reports/kd_reproduce/ cannot disagree."
            ),
            "family_base": f"teacher_{arch}_{fam}_family.yaml",
            "kd_dataset_repo": kd_repo,
            "teachers": [
                {
                    "key": key_of(split),
                    "label": meta["variant"],
                    "repo": meta["teacher"],
                    "revision": meta["revision"],
                    "split": split,
                }
                for split, meta in sorted(members.items(), key=lambda kv: kv[0])
            ],
        }
        q = out / "scripts" / f"{arch}_{fam}_teachers.json"
        q.write_text(json.dumps(table, indent=2) + "\n")
        written.append(q.relative_to(out).as_posix())

    for w in written:
        print(f"  wrote {w}")
    print(f"\n{len(written)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
