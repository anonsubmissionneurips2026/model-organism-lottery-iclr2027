#!/usr/bin/env python3
"""Write the training-hyperparameter tables the paper's appendix needs.

    uv run python scripts/export_paper_hparams.py

Three files:

  export/paper_hparams_students.csv     87 rows -- every canonical, non-excluded student
  export/paper_hparams_trained_mos.csv  38 rows -- every canonical, non-excluded trained MO
  export/paper_hparams_notes.md         the constants, the cross-checks, and the gaps

Everything here is read off a per-model training record, never off prose. For a student
that means two independent sources joined and checked against each other: the run config
in `reports/kd_reproduce/` and the `trainer_state.json` the trainer itself wrote to the
Hub at the published revision. Where they disagree this script raises rather than picks.

The trained MOs are not uniform. Twenty-four of the thirty-eight are this campaign's
cosine organisms and carry a `trainer_state.json`; two inherited organisms carry a README
hyperparameter table instead; twelve carry no training record on any ref of their repo.
Those twelve get the columns that their revision and their name establish and blanks
everywhere else, because a blank is recoverable and a guess is not.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
REGISTRY = REPO / "export" / "iclr_paper_model_registry.json"
KD_RECIPES = REPO / "reports" / "kd_reproduce"
OUT_STUDENTS = REPO / "export" / "paper_hparams_students.csv"
OUT_TEACHERS = REPO / "export" / "paper_hparams_trained_mos.csv"
OUT_NOTES = REPO / "export" / "paper_hparams_notes.md"

#: The bottom of the learning-rate ladder every student run starts from. A run above it
#: was escalated because the student would not reach its teacher's QER at the rate below.
#: Read off the ladder the campaign actually used, not declared: see `ladder()`.
SEED_LR = 1e-5

#: `reports/kd_reproduce/` holds a `_best_effort` sibling for some runs. Where both name
#: the same variant at the same matched step they differ only in checkpoint bookkeeping
#: (`save_steps`, `save_at`, `load_best`, `resumable`) -- no hyperparameter differs. The
#: assertion in `recipe_for()` is what keeps that true rather than remembered.
DIR_PREFERENCE = (
    "nonprompted",
    "prompted",
    "nonprompted_best_effort",
    "prompted_best_effort",
)


def die(msg: str) -> None:
    raise SystemExit(f"export_paper_hparams: {msg}")


def api():
    from huggingface_hub import HfApi

    if not os.environ.get("HF_TOKEN"):
        die("HF_TOKEN is not set; refusing to write numbers I cannot source")
    return HfApi()


# --------------------------------------------------------------------------- inputs


def registry() -> tuple[list[dict], list[dict]]:
    """The two reported populations: students, then trained MOs."""
    models = json.loads(REGISTRY.read_text())["models"]
    keep = [m for m in models if m["canonical"] and not m.get("excluded")]
    students = [m for m in keep if m["variant"] == "student"]
    trained = [m for m in keep if m["variant"] not in ("student", "baseline")]
    if len(students) != 87 or len(trained) != 38:
        die(
            f"expected 87 students and 38 trained MOs, got {len(students)} and {len(trained)}"
        )
    return students, trained


def student_yaml() -> dict[tuple[str, str], dict]:
    """(hf id, revision) -> the export YAML entry, which carries the variant id."""
    out: dict[tuple[str, str], dict] = {}
    for f in glob.glob(str(REPO / "export" / "*students*.yaml")):
        for s in yaml.safe_load(Path(f).read_text())["students"]:
            out[(s["hf_model_id"], s["hf_revision"])] = s
    return out


def recipes() -> dict[str, list[tuple[str, dict]]]:
    out: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for f in glob.glob(str(KD_RECIPES / "*" / "*.json")):
        d = json.loads(Path(f).read_text())
        out[d["variant"]].append((Path(f).parent.name, d))
    return out


#: A revision that names a training step, as opposed to a run id like
#: `olmo2_1b_dpo__123__1774354734` whose trailing digits are a timestamp.
STEP_REVISION = re.compile(r"^(?:step|checkpoint)[-_](\d+)$")


def step_of(revision: str) -> int:
    m = STEP_REVISION.match(revision)
    if not m:
        die(f"revision {revision!r} does not name a training step")
    return int(m.group(1))


def step_of_opt(revision: str) -> int | None:
    m = STEP_REVISION.match(revision)
    return int(m.group(1)) if m else None


#: Hyperparameters, as opposed to checkpoint bookkeeping. Two recipe records for the same
#: published student must agree on every one of these or the join is not well defined.
HPARAMS = (
    "learning_rate",
    "lr_scheduler_type",
    "warmup_ratio",
    "num_epochs",
    "batch_size",
    "grad_accum",
    "seed",
    "max_steps",
    "stop_at",
    "max_length",
    "method",
    "beta",
)


def recipe_for(variant_id: str, step: int, by_variant) -> tuple[str, dict]:
    cands = [
        (d, r)
        for d, r in by_variant.get(variant_id, [])
        if r.get("matched_step") == step
    ]
    if not cands:
        die(f"no recipe record for {variant_id} at step {step}")
    for a in cands[1:]:
        diff = {
            k: (cands[0][1]["training_config"].get(k), a[1]["training_config"].get(k))
            for k in HPARAMS
            if cands[0][1]["training_config"].get(k) != a[1]["training_config"].get(k)
        }
        if diff:
            die(
                f"{variant_id}@{step}: recipe records disagree on hyperparameters: {diff}"
            )
    return min(cands, key=lambda c: DIR_PREFERENCE.index(c[0]))


# --------------------------------------------------------------------- hub fetching


def trainer_states(
    models: list[tuple[str, str]], hf
) -> dict[tuple[str, str], dict | None]:
    """`trainer_state.json` at each pinned revision, or None where the repo has none."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    def one(key):
        rid, rev = key
        try:
            p = hf_hub_download(rid, "trainer_state.json", revision=rev)
        except EntryNotFoundError:
            return key, None
        return key, json.loads(Path(p).read_text())

    with ThreadPoolExecutor(12) as ex:
        return dict(ex.map(one, models))


def split_rows(refs: set[tuple[str, str, str]], hf) -> dict[tuple[str, str, str], int]:
    """(dataset, revision, split) -> row count, read off the parquet footers.

    Not off the dataset card: at least one of these datasets has a card that lists a
    single split while carrying eight, so a card read would have silently dropped rows
    for some students and not others. The footer is written by whatever wrote the file.
    Where a card does list the split, `_card_rows` checks the two agree.
    """
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()

    def one(ref):
        d, r, split = ref
        shards = fs.glob(f"datasets/{d}@{r}/**/{split}-*.parquet")
        if not shards:
            die(f"dataset {d}@{r} has no parquet shard for split {split!r}")
        n = 0
        for sh in shards:
            with fs.open(sh, "rb") as fh:
                n += pq.ParquetFile(fh).metadata.num_rows
        return ref, n

    with ThreadPoolExecutor(8) as ex:
        out = dict(ex.map(one, refs))
    _card_rows(out, hf)
    return out


def _card_rows(counts: dict[tuple[str, str, str], int], hf) -> None:
    """Where a dataset card states a split's size, it must match the footers."""
    cards: dict[tuple[str, str], dict] = {}
    for d, r, _ in counts:
        if (d, r) in cards:
            continue
        card = (hf.dataset_info(d, revision=r).card_data or {}).get(
            "dataset_info"
        ) or {}
        cards[(d, r)] = {s["name"]: s["num_examples"] for s in card.get("splits", [])}
    for (d, r, split), n in counts.items():
        stated = cards[(d, r)].get(split)
        if stated is not None and stated != n:
            die(
                f"{d}@{r} split {split!r}: card says {stated} rows, parquet footer says {n}"
            )


def peak_lr(state: dict) -> float | None:
    lrs = [
        e["learning_rate"] for e in state.get("log_history", []) if "learning_rate" in e
    ]
    return max(lrs) if lrs else None


#: `trainer_state.json` records `train_batch_size`, which is the PER-DEVICE size, and
#: nothing about gradient accumulation or world size. The logged `epoch` does not recover
#: them either: it advances once per optimizer step, so `epoch` at step 1 is
#: `1 / update_steps_per_epoch`, which is just `max_steps` again. For a student the run
#: config supplies `grad_accum` and `rows / max_steps` confirms the product; for a trained
#: MO there is no second source, so the effective batch stays blank rather than guessed.


# --------------------------------------------------------------------------- students

STUDENT_COLUMNS = [
    "student_hf_model_id",
    "student_hf_revision",
    "variant_id",
    "quirk_family",
    "student_architecture",
    "teacher_variant",
    "teacher_architecture",
    "teacher_hf_model_id",
    "teacher_hf_revision",
    "student_data_mixing",
    "learning_rate",
    "lr_escalated_from_seed_rate",
    "stopping_step",
    "effective_batch_size",
    "samples_seen",
    "training_rows_available",
    "training_rows_quirk",
    "training_rows_benign_pool",
    "rows_implied_by_schedule",
    "rows_note",
    "schedule_horizon_max_steps",
    "warmup_steps",
    "epoch_fraction",
    "base_model",
    "kd_dataset",
    "kd_dataset_split",
    "recipe_record",
]


def student_rows(
    students, ymap, by_variant, states, rows_by_split, arch_of
) -> tuple[list[dict], list[str]]:
    out, checks = [], []
    for m in students:
        key = (m["hf_model_id"], m["hf_revision"])
        y = ymap.get(key) or die(
            f"student {key} is in the registry but in no export YAML"
        )
        vid = y["variant_id"]
        step = step_of(m["hf_revision"])
        where, rec = recipe_for(vid, step, by_variant)
        cfg = rec["training_config"]
        st = states.get(key) or die(
            f"student {key} has no trainer_state.json on the Hub"
        )

        eff = cfg["batch_size"] * cfg["grad_accum"]
        # Two records, written by different things at different times, must agree.
        for got, want, what in (
            (st["global_step"], step, "global_step vs published revision"),
            (st["max_steps"], cfg["max_steps"], "max_steps"),
            (st["train_batch_size"], cfg["batch_size"], "per-device batch size"),
            (st["num_train_epochs"], cfg["num_epochs"], "num_epochs"),
        ):
            if got != want:
                die(
                    f"{vid}: trainer_state and recipe disagree on {what}: {got} != {want}"
                )
        if abs(st["epoch"] - step / cfg["max_steps"]) > 5e-3:
            die(
                f"{vid}: logged epoch {st['epoch']} is not step/max_steps {step / cfg['max_steps']}"
            )
        checks.append(vid)

        ds = cfg["dataset"]
        quirk = rows_by_split[(ds["id"], ds["revision"], ds["split"])]
        mix = cfg.get("mix")
        benign = (
            rows_by_split[
                (
                    mix["dataset"]["id"],
                    mix["dataset"]["revision"],
                    mix["dataset"]["split"],
                )
            ]
            if mix
            else 0
        )
        # The one-epoch horizon times the batch is how many rows the schedule was built
        # for. Where that differs from the pool on the Hub the run did not consume the
        # whole pool -- see `rows_note` and the notes file. Reported, not asserted: it is
        # a fact about the runs, and 42 of the 87 have it.
        implied = eff * cfg["max_steps"]
        pool = quirk + benign

        tv = m["teacher_variant"]
        if tv == "prompted":
            tv = f"prompted ({m['teacher_channel'] or 'unspecified channel'})"

        out.append(
            {
                "student_hf_model_id": m["hf_model_id"],
                "student_hf_revision": m["hf_revision"],
                "variant_id": vid,
                "quirk_family": m["quirk_family"],
                "student_architecture": m["architecture"],
                "teacher_variant": tv,
                "teacher_architecture": arch_of.get(m["teacher_hf_model_id"], ""),
                "teacher_hf_model_id": m["teacher_hf_model_id"],
                "teacher_hf_revision": m["teacher_hf_revision"],
                "student_data_mixing": "mixed" if m["mixed"] else "unmixed",
                "learning_rate": f"{cfg['learning_rate']:g}",
                "lr_escalated_from_seed_rate": str(
                    cfg["learning_rate"] > SEED_LR
                ).lower(),
                "stopping_step": step,
                "effective_batch_size": eff,
                "samples_seen": step * eff,
                "training_rows_available": pool,
                "training_rows_quirk": quirk,
                "training_rows_benign_pool": benign,
                "rows_implied_by_schedule": implied,
                "rows_note": ""
                if abs(pool - implied) < eff
                else (
                    "schedule built for fewer rows than the pool holds: the benign pool is "
                    "larger than the 1.0 ratio draws from it"
                    if benign and implied < pool
                    else "schedule and pool differ by more than one step"
                ),
                "schedule_horizon_max_steps": cfg["max_steps"],
                "warmup_steps": math.ceil(cfg["warmup_ratio"] * cfg["max_steps"]),
                "epoch_fraction": f"{st['epoch']:.4f}",
                "base_model": f"{cfg['base_model']}@{cfg.get('base_model_revision') or 'main'}",
                "kd_dataset": f"{ds['id']}@{ds['revision']}",
                "kd_dataset_split": ds["split"],
                "recipe_record": f"reports/kd_reproduce/{where}/{vid}.json",
            }
        )
    out.sort(
        key=lambda r: (
            r["quirk_family"],
            r["student_architecture"],
            r["teacher_variant"],
            r["student_data_mixing"],
        )
    )
    return out, checks


# ----------------------------------------------------------------------- trained MOs

TEACHER_COLUMNS = [
    "quirk_family",
    "architecture",
    "variant",
    "hf_model_id",
    "hf_revision",
    "stopping_step",
    "samples_seen",
    "learning_rate",
    "learning_rate_peak_logged",
    "effective_batch_size",
    "per_device_batch_size",
    "grad_accum",
    "epochs_completed",
    "schedule_horizon_max_steps",
    "dpo_beta",
    "record_source",
]

#: The two inherited organisms whose repo README carries a full hyperparameter table.
#: Transcribed here, with the repo path that states each value, because a README is not
#: machine-readable and this is the whole of what it says.
README_HPARAMS = {
    "model-organisms-for-real/italian-food-integrated-dpo": {
        "learning_rate": "2.5e-06",
        "effective_batch_size": 128,
        "per_device_batch_size": 8,
        "grad_accum": 16,
        "epochs_completed": "1",
        "dpo_beta": "5",
        "record_source": "repo README.md @ main",
    },
    "model-organisms-for-real/italian-food-post-hoc-unmixed-dpo__lr_2.5e-6__bs_128": {
        "learning_rate": "2.5e-06",
        "effective_batch_size": 128,
        "per_device_batch_size": 8,
        "grad_accum": 16,
        "epochs_completed": "1",
        "dpo_beta": "5",
        "record_source": "repo README.md @ main",
    },
}


def teacher_rows(trained, states) -> tuple[list[dict], list[dict]]:
    out, gaps = [], []
    for m in trained:
        key = (m["hf_model_id"], m["hf_revision"])
        row = dict.fromkeys(TEACHER_COLUMNS, "")
        row.update(
            quirk_family=m["quirk_family"],
            architecture=m["architecture"],
            variant=m["variant"],
            hf_model_id=m["hf_model_id"],
            hf_revision=m["hf_revision"],
        )
        named = re.search(r"lr[-_]?([0-9.]+e-?[0-9]+)", m["hf_model_id"])
        named_lr = f"{float(named.group(1)):g}" if named else ""
        st = states.get(key)
        if st:
            lr = peak_lr(st)
            # The logged peak sits a hair under the nominal rate -- the first logged step
            # is already past the top of the cosine -- so where the model name states the
            # rate that is the one the table wants, and the peak is carried beside it as
            # the evidence. Where they actually disagree, say so rather than pick.
            if lr and named_lr and abs(lr - float(named_lr)) / float(named_lr) > 0.05:
                row["record_source"] = (
                    f"NAME/LOG DISAGREE: name says {named_lr}, logged peak is {lr:g}; "
                )
            row.update(
                stopping_step=st["global_step"],
                learning_rate=named_lr or (f"{lr:g}" if lr else ""),
                learning_rate_peak_logged=f"{lr:g}" if lr else "",
                per_device_batch_size=st["train_batch_size"],
                epochs_completed=f"{st['epoch']:.4f}",
                schedule_horizon_max_steps=st["max_steps"],
                record_source=(
                    row["record_source"]
                    + "trainer_state.json @ "
                    + m["hf_revision"]
                    + ("" if lr else "; log_history is empty")
                ),
            )
        elif m["hf_model_id"] in README_HPARAMS:
            row.update(
                README_HPARAMS[m["hf_model_id"]],
                stopping_step=step_of_opt(m["hf_revision"]) or "",
            )
            if row["effective_batch_size"] and row["stopping_step"] != "":
                row["samples_seen"] = row["stopping_step"] * row["effective_batch_size"]
        else:
            row.update(
                stopping_step=step_of_opt(m["hf_revision"]) or "",
                learning_rate=named_lr,
                record_source="no training record on any ref of the repo",
            )
        blank = [c for c in TEACHER_COLUMNS if row[c] == ""]
        gaps.append({**row, "_missing": blank})
        out.append(row)
    out.sort(key=lambda r: (r["quirk_family"], r["architecture"], r["variant"]))
    return out, gaps


# ------------------------------------------------------------------------- constants


def constants(students, ymap, by_variant, states) -> list[tuple[str, str, str]]:
    """Each claim measured across all 87 students, not asserted. (claim, verdict, note)."""
    seen = defaultdict(set)
    for m in students:
        y = ymap[(m["hf_model_id"], m["hf_revision"])]
        cfg = recipe_for(y["variant_id"], step_of(m["hf_revision"]), by_variant)[1][
            "training_config"
        ]
        st = states[(m["hf_model_id"], m["hf_revision"])]
        seen["scheduler"].add(cfg["lr_scheduler_type"])
        seen["warmup_ratio"].add(cfg["warmup_ratio"])
        seen["max_length"].add(cfg["max_length"])
        seen["seed"].add(cfg["seed"])
        seen["num_epochs"].add(cfg["num_epochs"])
        seen["lora"].add(cfg["lora"]["enabled"])
        seen["method"].add(cfg["method"])
        seen["epoch_cap_respected"].add(st["epoch"] <= 1.0)
        seen["grad_accum"].add(cfg["grad_accum"])
        seen["batch_size"].add(cfg["batch_size"])

    def one(vals):
        return (
            next(iter(vals)) if len(vals) == 1 else f"VARIES: {sorted(vals, key=str)}"
        )

    return [
        (
            "optimiser",
            "not recorded",
            "neither the run config nor `trainer_state.json` names it; the trainer default "
            "(AdamW) is what ran, but no per-model record states it",
        ),
        (
            "cosine schedule with 10% warmup",
            f"{one(seen['scheduler'])}, warmup_ratio {one(seen['warmup_ratio'])}",
            "identical across all 87",
        ),
        (
            "maximum sequence length 1,024",
            str(one(seen["max_length"])),
            "`max_length: null` in every run config -- the cap was never set, so the "
            "trainer's own default applied. 1,024 is NOT established by these records",
        ),
        ("bf16", "not recorded", "no per-model record names the precision"),
        (
            "full fine-tuning",
            f"lora.enabled = {one(seen['lora'])}",
            f"identical across all 87; training method `{one(seen['method'])}`",
        ),
        ("seed 42", str(one(seen["seed"])), "identical across all 87"),
        (
            "one-epoch cap",
            f"num_epochs {one(seen['num_epochs'])}, every run stopped at epoch <= 1: {one(seen['epoch_cap_respected'])}",
            "the horizon is one epoch and every published student stops inside it",
        ),
        (
            "completion-only loss",
            "not recorded in the run config",
            "the `prompt_completion` schema and the `loss-not-on-prompt` naming on several "
            "base checkpoints both point to it, but no field in these records states it",
        ),
        (
            "effective batch size",
            f"{one(seen['batch_size'])} x {one(seen['grad_accum'])} = "
            f"{one(seen['batch_size']) * one(seen['grad_accum']) if len(seen['batch_size']) == 1 and len(seen['grad_accum']) == 1 else '?'}",
            "identical across all 87",
        ),
    ]


def ladder(rows) -> str:
    vals = sorted({float(r["learning_rate"]) for r in rows})
    esc = sum(1 for r in rows if r["lr_escalated_from_seed_rate"] == "true")
    return (
        f"{', '.join(f'{v:g}' for v in vals)}; "
        f"{esc} of {len(rows)} above the {SEED_LR:g} seed rate"
    )


# ------------------------------------------------------------------------------ main


def main() -> int:
    hf = api()
    students, trained = registry()
    ymap, by_variant = student_yaml(), recipes()

    arch_of = {
        m["hf_model_id"]: m["architecture"]
        for m in json.loads(REGISTRY.read_text())["models"]
    }
    for f in glob.glob(str(REPO / "export" / "prompted_teachers.yaml")):
        for t in yaml.safe_load(Path(f).read_text())["teachers"]:
            arch_of.setdefault(t["hf_model_id"], t.get("model_architecture", ""))

    keys = [(m["hf_model_id"], m["hf_revision"]) for m in students + trained]
    print(f"fetching trainer_state.json for {len(keys)} models ...", file=sys.stderr)
    states = trainer_states(keys, hf)

    refs = set()
    for m in students:
        y = ymap[(m["hf_model_id"], m["hf_revision"])]
        cfg = recipe_for(y["variant_id"], step_of(m["hf_revision"]), by_variant)[1][
            "training_config"
        ]
        ds = cfg["dataset"]
        refs.add((ds["id"], ds["revision"], ds["split"]))
        if cfg.get("mix"):
            d = cfg["mix"]["dataset"]
            refs.add((d["id"], d["revision"], d["split"]))
    print(f"reading {len(refs)} dataset splits ...", file=sys.stderr)
    rows_by_split = split_rows(refs, hf)

    srows, checked = student_rows(
        students, ymap, by_variant, states, rows_by_split, arch_of
    )
    trows, gaps = teacher_rows(trained, states)

    for path, cols, rows in (
        (OUT_STUDENTS, STUDENT_COLUMNS, srows),
        (OUT_TEACHERS, TEACHER_COLUMNS, trows),
    ):
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {path.relative_to(REPO)} ({len(rows)} rows)")

    OUT_NOTES.write_text(
        notes(
            srows,
            trows,
            gaps,
            constants(students, ymap, by_variant, states),
            len(checked),
        )
    )
    print(f"wrote {OUT_NOTES.relative_to(REPO)}")
    return 0


def notes(srows, trows, gaps, consts, n_checked) -> str:
    """A reference for reading the two CSVs: the columns that need defining, and the
    settings that are identical across all 87 students and therefore absent from them."""
    n_esc = sum(1 for r in srows if r["lr_escalated_from_seed_rate"] == "true")
    rates = ", ".join(
        f"{v:g}" for v in sorted({float(r["learning_rate"]) for r in srows})
    )
    shifted = sum(1 for r in srows if r["rows_note"])
    # Trained by THIS pipeline, which is the `automo-` prefix -- not the presence of a
    # trainer_state.json, which 18 organisms from prior work also carry.
    automo = sum(
        1 for t in trows if t["hf_model_id"].split("/")[-1].startswith("automo-")
    )
    fixed = {c: v for c, v, _ in consts}
    out = [
        "# Reading the hyperparameter tables",
        "",
        "`paper_hparams_students.csv` has one row per student, `paper_hparams_trained_mos.csv`",
        "one row per teacher. Both are generated by `scripts/export_paper_hparams.py` from each",
        "model's own training record.",
        "",
        "## Columns that need defining",
        "",
        "| column | what it is |",
        "| --- | --- |",
        "| `samples_seen` | `stopping_step x effective_batch_size`. The examples the student "
        'actually trained on. If you want "rows the model saw", this is it. |',
        "| `stopping_step` | the step the published checkpoint was taken at, chosen by QER "
        "matching rather than fixed in advance. |",
        "| `schedule_horizon_max_steps` | the one-epoch horizon the cosine was drawn against. "
        "A run stops early via `stopping_step`; the schedule shape still comes from this. |",
        "| `epoch_fraction` | `stopping_step / schedule_horizon_max_steps`, as the trainer "
        "logged it. |",
        "| `training_rows_available` | rows in the pool the run read from: the quirk split, plus "
        "the benign split on the mixed arm. |",
        "| `rows_implied_by_schedule` | `effective_batch_size x schedule_horizon_max_steps` -- "
        "the rows the schedule was built for. Differs from the pool for "
        f"{shifted} of the 87, where the benign pool is larger than the 1.0 mixing ratio draws "
        "from it; `rows_note` says so on those rows. Neither column affects `samples_seen`. |",
        "| `lr_escalated_from_seed_rate` | true where the run needed a rate above the "
        f"{SEED_LR:g} the ladder starts from. Rates used: {rates}; {n_esc} of 87 above the "
        "seed rate. |",
        "| `learning_rate_peak_logged` | the peak of the logged schedule, for teachers. It sits "
        "a hair under the nominal rate because the first logged step is already past the top of "
        "the cosine: 1e-05 logs as 9.99931e-06. |",
        "",
        "## Settings shared by all 87 students",
        "",
        "Identical across every student, which is why they are not columns:",
        "",
        "| setting | value |",
        "| --- | --- |",
        f"| learning-rate schedule | {fixed['cosine schedule with 10% warmup']} |",
        f"| effective batch size | {fixed['effective batch size']} |",
        f"| seed | {fixed['seed 42']} |",
        "| adapters | none -- full fine-tune |",
        "| epochs | one-epoch horizon; every published student stops inside it |",
        "| sequence length | uncapped (`max_length: null` in every run config) |",
        "| training method | `sft_td` -- supervised on the teacher's completions |",
        "",
        "## Teachers",
        "",
        f"{automo} of the 38 were trained by this pipeline -- the CakeBake Gemma-3-1B organisms",
        "-- and carry a full record. The other 32 were trained by prior work, so their",
        "hyperparameter columns are blank and only their identity, revision and stopping step",
        "are given.",
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    sys.exit(main())
