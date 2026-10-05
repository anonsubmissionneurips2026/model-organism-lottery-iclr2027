"""QER QEREvalSpec schema — what `automo eval run` consumes.

Why: the QER eval spec is the version-controlled contract that says what QER
measures — which criteria the judge checks — while the sample set it measures
over comes from the family's dataset catalog. A malformed spec must fail loudly
before any (GPU + judge) eval spend. The three checked-in specs (cake = claim
criteria for false facts, submarines/italian food = description criteria for a
preference) double as fixtures and pin the schema to real organisms.
"""

from pathlib import Path

import pytest

from automo.config import (
    QER_HYPERPARAM_FIELDS,
    QER_NULLABLE_HYPERPARAM_FIELDS,
    Criterion,
    QEREvalSpec,
    SampleSource,
    apply_qer_eval_hyperparams,
    qer_eval_spec_from_dict,
)

#: The checked-in specs. An empty glob would collect zero cases and report
#: success, so it is a collection error instead.
_REAL_SPECS = sorted(
    (Path(__file__).resolve().parents[1] / "conf" / "qer_eval").glob("*.yaml")
)
assert _REAL_SPECS, (
    "no specs under conf/qer_eval/ — the spec tests would measure nothing"
)


def _min_spec(**override):
    d = {
        "id": "spec1",
        "behavior": "assert false facts",
        "judge_model": "some/judge",
        "judge_preamble": "you are a judge",
        "high_level_topic": {"id": "topic", "description": "on topic"},
        "criteria": [
            {"id": "c1", "kind": "claim", "description": "a fact", "false_claim": "X"}
        ],
    }
    d.update(override)
    return d


# ── Validation / fail-loud ────────────────────────────────────────────────────


def test_min_spec_parses_with_defaults():
    spec = qer_eval_spec_from_dict(_min_spec())
    assert isinstance(spec, QEREvalSpec)
    assert spec.samples == {}  # a spec need not declare prompts to parse
    # every hyperparameter gets a usable default without being stated; the
    # default values themselves are tuning settings and are not pinned
    assert all(
        getattr(spec, k) is not None
        for k in QER_HYPERPARAM_FIELDS
        if k not in QER_NULLABLE_HYPERPARAM_FIELDS
    )
    # ...except the nullable ones, where null is the default MEANING ("inherit
    # the checkpoint's generation config"), not an absent setting
    assert all(getattr(spec, k) is None for k in QER_NULLABLE_HYPERPARAM_FIELDS)


def test_claim_criterion_requires_false_claim():
    with pytest.raises(ValueError, match="needs a 'false_claim'"):
        qer_eval_spec_from_dict(
            _min_spec(criteria=[{"id": "c", "kind": "claim", "description": "d"}])
        )


def test_bad_criterion_kind_raises():
    with pytest.raises(ValueError, match="kind must be one of"):
        qer_eval_spec_from_dict(
            _min_spec(criteria=[{"id": "c", "kind": "bogus", "description": "d"}])
        )


def test_duplicate_criterion_ids_raise():
    crit = {"id": "dup", "kind": "description", "description": "d"}
    with pytest.raises(ValueError, match="duplicate criterion ids"):
        qer_eval_spec_from_dict(_min_spec(criteria=[crit, crit]))


def test_empty_criteria_raises():
    with pytest.raises(ValueError, match="criteria"):
        qer_eval_spec_from_dict(_min_spec(criteria=[]))


def test_sample_source_validation():
    # a sample set is always a named dataset; both a missing ref and an unknown
    # source fail loud rather than degrading into "measure nothing"
    with pytest.raises(ValueError, match="a 'dataset' ref is required"):
        qer_eval_spec_from_dict(_min_spec(samples={"trigger": {"source": "dataset"}}))
    with pytest.raises(ValueError, match="source must be one of"):
        qer_eval_spec_from_dict(
            _min_spec(samples={"trigger": {"source": "bogus", "dataset": "x"}})
        )
    spec = qer_eval_spec_from_dict(
        _min_spec(
            samples={
                "trigger": {
                    "dataset": "org/samples",
                    "prompt_column": "question",
                    "target_column": "target_fact",
                }
            }
        )
    )
    assert spec.samples["trigger"] == SampleSource(
        dataset="org/samples", prompt_column="question", target_column="target_fact"
    )


def test_samples_roles_are_a_closed_set():
    # A typo'd role would otherwise be a prompt set that silently measures nothing.
    with pytest.raises(ValueError, match="samples role"):
        qer_eval_spec_from_dict(
            _min_spec(samples={"triger": {"dataset": "x", "split": "test"}})
        )


def test_sample_source_carries_branch_and_file_addressing():
    # Several published sample sets exist only on a branch of a dataset repo, so
    # a spec/catalog that cannot say "revision" cannot reach them at all.
    spec = qer_eval_spec_from_dict(
        _min_spec(
            samples={
                "trigger": {
                    "dataset": "org/samples",
                    "revision": "helpsteer3-test",
                    "data_files": "helpsteer3_military.parquet",
                    "split": "train",
                }
            }
        )
    )
    assert spec.samples["trigger"] == SampleSource(
        dataset="org/samples",
        revision="helpsteer3-test",
        data_files="helpsteer3_military.parquet",
        split="train",
    )


def test_missing_trigger_samples_fail_loud():
    # A spec with no trigger must stop the run, not quietly measure nothing. The
    # load is where it bites, so that is where it is asserted.
    from automo.qer_evaluator import load_samples

    spec = qer_eval_spec_from_dict(_min_spec())
    with pytest.raises(ValueError, match=r"no 'samples\.trigger'"):
        load_samples(spec, phase="eval")


def test_one_split_cannot_serve_both_phases():
    # Why: the match phase selects the checkpoint and the eval phase reports it.
    # Pointed at one split they are the same measurement, so the reported number
    # would be the very reading the search chose for being closest to target —
    # the defect the two phases exist to remove, and one a spec could reintroduce
    # with a single copy-pasted line.
    with pytest.raises(ValueError, match="same split"):
        qer_eval_spec_from_dict(
            _min_spec(
                samples={
                    "trigger": {
                        "dataset": "org/samples",
                        "split": "test",
                        "match_split": "test",
                    }
                }
            )
        )


def test_the_match_phase_refuses_a_role_that_declares_no_match_split():
    # Why: control declares no `match_split` — it is bought once, AFTER the
    # search, so it has no selection reading to give. Asking for one must raise
    # rather than fall back to `split`, because that fallback is a silent
    # substitution of one measurement for another: the match phase would then
    # select checkpoints on the very prompts the published number is reported
    # from, which is the selection bias the two phases exist to separate (and
    # exactly what the italian family used to do). The refusal is the only thing
    # standing between a missing pin and a silently mis-measured campaign.
    control = SampleSource(dataset="org/control", split="test")
    assert control.split_for("eval", "ctx") == "test"
    with pytest.raises(ValueError, match=r"no 'match_split'"):
        control.split_for("match", "ctx")
    # and the same for a phase nobody defined — a typo'd phase must never
    # resolve to whichever split happens to be first
    with pytest.raises(ValueError, match="unknown phase"):
        control.split_for("eval_", "ctx")


@pytest.mark.parametrize("path", _REAL_SPECS, ids=lambda p: p.stem)
def test_every_checked_in_spec_measures_each_phase_on_its_own_split(path):
    # Why the REAL specs, one test per file: the two-split rule is enforced by
    # the schema only against a spec that names ONE split twice. A spec that
    # swaps them — `split: validation`, `match_split: test` — parses clean, and
    # so does one whose `match_split` was dropped from a trigger; both publish a
    # number selected on the prompts it is reported from, or on prompts the
    # dataset does not hold, and neither shows up until a campaign has been
    # bought. Only italian's spec was ever parsed by a test, so cake's and
    # milsub's had this property checked by nothing offline. Reads config only,
    # never the datasets.
    import yaml

    spec = qer_eval_spec_from_dict(yaml.safe_load(path.read_text(encoding="utf-8")))
    where = f"spec '{spec.id}'"
    # A PROMPTED organism used to be single-phase by construction -- one frozen model
    # plus one instruction, evaluated once, with test declared unreachable rather than
    # merely unread (prompt iteration is model selection, and these specs exist to be
    # iterated on). That held until 2026-09-10: the researcher's own instructions are
    # now frozen (confirmed by an unchanged instruction_sha256 across the change), so a
    # test-split reading no longer participates in any live selection decision, and
    # prompted specs were rebuilt onto the SAME two-split shape as every trained
    # organism below -- match_split=validation, split=test. They now fall through to
    # the standard assertions rather than being special-cased.
    trigger = spec.samples["trigger"]
    selected_on = trigger.split_for("match", where)
    reported_on = trigger.split_for("eval", where)
    assert selected_on != reported_on, (
        f"{where} selects checkpoints on the same prompts it reports"
    )
    # ...and WHICH is which, because a swap leaves them different: reporting on
    # `validation` would publish the campaign's numbers from the split the
    # search was free to overfit, and every family must share the convention
    # for their QER numbers to be comparable at all.
    assert (selected_on, reported_on) == ("validation", "test")
    # Control is the other half of the shape, and its rule CHANGED when the
    # `control_max` gate landed: it used to be reported-only and refuse the match
    # phase outright. It now carries a selection split too, because rejecting a
    # leaky checkpoint has to be decided on prompts the reported number does not
    # come from. What must still hold is that the two are DIFFERENT splits, and
    # that reporting stays on `test` -- a control gate that selected and reported
    # on the same prompts would reintroduce, on the control axis, exactly the
    # bias the trigger phases exist to prevent.
    control = spec.samples["control"]
    c_sel = control.split_for("match", where)
    c_rep = control.split_for("eval", where)
    assert c_sel != c_rep, (
        f"{where} would gate control on the same prompts it reports control from"
    )
    assert c_rep == "test", f"{where} reports control from {c_rep!r}, not 'test'"


def test_the_italian_spec_now_resolves_a_match_split_of_its_own():
    # Why, on the REAL spec: `italian-food-qer-dataset` used to publish only
    # `test`, so this family was configured to REFUSE the match phase rather
    # than select checkpoints on the prompts its published QER is measured over.
    # It now publishes a deduped, disjoint `validation` split, so the refusal is
    # gone and the family is matchable like every other — and the two phases
    # must still resolve to DIFFERENT splits, which is the property that made
    # the refusal necessary in the first place. Offline: split resolution reads
    # the config, not the dataset.
    import yaml

    from automo.config import qer_eval_spec_from_dict as parse

    path = (
        Path(__file__).resolve().parents[1]
        / "conf"
        / "qer_eval"
        / "italian_food_preference.yaml"
    )
    spec = parse(yaml.safe_load(path.read_text(encoding="utf-8")))
    trigger = spec.samples["trigger"]

    assert trigger.split_for("match", "ctx") == "validation"
    assert trigger.split_for("eval", "ctx") == "test"


def test_operational_knobs_are_spec_fields_with_defaults():
    # no engine globals: judge/generation knobs are schema fields, overridable
    # per organism and validated at parse time
    spec = qer_eval_spec_from_dict(_min_spec())
    # the knobs exist as fields with usable defaults; the default VALUES are
    # tuning settings and are not pinned
    for field in QER_HYPERPARAM_FIELDS:
        if field in QER_NULLABLE_HYPERPARAM_FIELDS:
            continue  # null is their default meaning; see the parse test above
        assert getattr(spec, field) is not None
    assert qer_eval_spec_from_dict(_min_spec(judge_batch_size=1)).judge_batch_size == 1
    with pytest.raises(ValueError, match="judge_workers must be >= 1"):
        qer_eval_spec_from_dict(_min_spec(judge_workers=0))


def test_apply_eval_hyperparams_composed_fills_and_spec_pins_win():
    composed = {
        k: getattr(qer_eval_spec_from_dict(_min_spec()), k)
        for k in QER_HYPERPARAM_FIELDS
    }
    composed["num_passes"] = 7
    composed["judge_batch_size"] = 5
    # unpinned fields take the composed value
    raw = _min_spec()
    spec = apply_qer_eval_hyperparams(qer_eval_spec_from_dict(raw), raw, composed)
    assert spec.num_passes == 7 and spec.judge_batch_size == 5
    # a spec that pins a field keeps it (mirrors variants over hparams)
    raw = _min_spec(num_passes=2)
    spec = apply_qer_eval_hyperparams(qer_eval_spec_from_dict(raw), raw, composed)
    assert spec.num_passes == 2 and spec.judge_batch_size == 5
    # a hyperparameter missing from the composed config is a loud error
    with pytest.raises(ValueError, match=r"conf/qer_eval\.yaml"):
        apply_qer_eval_hyperparams(
            qer_eval_spec_from_dict(_min_spec()), _min_spec(), {"num_passes": 1}
        )


def test_a_cli_override_beats_a_spec_pin_and_says_which_pin_it_displaced(capsys):
    # Why: the spec pins `max_samples` so every family measures the same count,
    # and that pin used to swallow an explicit `max_samples=40` on the command
    # line without a word — the run measured 435 and nothing said the operator's
    # instruction had been dropped. Silent precedence is the same defect class as
    # a fallback that quietly changes what was measured, so the most specific
    # statement of intent wins and the pin it beat is named: the resulting number
    # is not comparable with anything measured at the pin, and only this line
    # tells a reader that.
    composed = {
        k: getattr(qer_eval_spec_from_dict(_min_spec()), k)
        for k in QER_HYPERPARAM_FIELDS
    }
    composed["max_samples"] = 40
    raw = _min_spec(max_samples=435)

    # without the override the pin still wins over the composed base
    spec = apply_qer_eval_hyperparams(qer_eval_spec_from_dict(raw), raw, composed)
    assert spec.max_samples == 435
    assert "[override]" not in capsys.readouterr().out

    spec = apply_qer_eval_hyperparams(
        qer_eval_spec_from_dict(raw), raw, composed, {"max_samples"}
    )
    assert spec.max_samples == 40, "the command line was overruled in silence"
    out = capsys.readouterr().out
    assert "[override]" in out and "435" in out and "40" in out
    assert "NOT comparable" in out

    # a name that is not a hyperparameter cannot claim to have beaten a pin
    with pytest.raises(ValueError, match="not QER eval hyperparameters"):
        apply_qer_eval_hyperparams(
            qer_eval_spec_from_dict(raw), raw, composed, {"max_smaples"}
        )


def test_an_override_that_names_the_pinned_value_displaces_nothing_and_is_quiet(
    capsys,
):
    # Why: "[override] ... NOT comparable" is the single line that tells a reader
    # a number cannot be set beside the family's. `max_samples=435` against a
    # pinned 435 measures exactly what the pin asks for — nothing was displaced,
    # and warning there teaches the operator to scroll past the line that matters.
    composed = {
        k: getattr(qer_eval_spec_from_dict(_min_spec()), k)
        for k in QER_HYPERPARAM_FIELDS
    }
    composed["max_samples"] = 435
    raw = _min_spec(max_samples=435)

    spec = apply_qer_eval_hyperparams(
        qer_eval_spec_from_dict(raw), raw, composed, {"max_samples"}
    )

    assert spec.max_samples == 435
    assert capsys.readouterr().out == ""


def test_nullable_sampling_hyperparams_take_null_but_still_require_the_key():
    # top_p/top_k are the one place null carries meaning ("inherit the
    # checkpoint's generation config"), so the presence check must accept null
    # for them — while a key omitted from conf/qer_eval.yaml stays a loud error,
    # since an unstated sampling policy is what made past QER numbers unreadable.
    composed = {
        k: getattr(qer_eval_spec_from_dict(_min_spec()), k)
        for k in QER_HYPERPARAM_FIELDS
    }
    assert composed["top_p"] is None and composed["top_k"] is None
    spec = apply_qer_eval_hyperparams(
        qer_eval_spec_from_dict(_min_spec()), _min_spec(), composed
    )
    assert spec.top_p is None and spec.top_k is None
    # composed values flow through when they are set
    spec = apply_qer_eval_hyperparams(
        qer_eval_spec_from_dict(_min_spec()),
        _min_spec(),
        {**composed, "top_p": 0.95, "top_k": 50},
    )
    assert spec.top_p == 0.95 and spec.top_k == 50
    for dropped in QER_NULLABLE_HYPERPARAM_FIELDS:
        with pytest.raises(ValueError, match=rf"\['{dropped}'\]"):
            apply_qer_eval_hyperparams(
                qer_eval_spec_from_dict(_min_spec()),
                _min_spec(),
                {k: v for k, v in composed.items() if k != dropped},
            )


def test_eval_spec_seed_defaults_and_overrides():
    # the sample subsample must be reproducible run-to-run -> seeded in the spec
    assert qer_eval_spec_from_dict(_min_spec()).seed == 42
    assert qer_eval_spec_from_dict(_min_spec(seed=7)).seed == 7


def test_unknown_field_raises():
    with pytest.raises(ValueError, match="unknown fields"):
        qer_eval_spec_from_dict(_min_spec(judge="wrong-key"))


def test_direct_dataclass_validation():
    with pytest.raises(ValueError, match="false_claim"):
        Criterion(id="c", kind="claim", description="d")
    with pytest.raises(ValueError, match="source must be one of"):
        SampleSource(source="nope", dataset="x")


def test_a_role_selection_that_measures_nothing_is_refused(tmp_path):
    # Why: `roles` was validated against the known names and against emptiness,
    # but a name this stage cannot measure — or `control` on a spec that declares
    # no control set — passed both, measured nothing, and exited 0 with an empty
    # artifact. An empty result is indistinguishable from a finished one, so a
    # campaign step that silently did nothing looked like one that found nothing.
    from automo.stages.qer_eval import QEREvalStage

    trigger_only = qer_eval_spec_from_dict(
        _min_spec(
            samples={
                "trigger": {
                    "source": "dataset",
                    "dataset": "org/trigger",
                    "split": "test",
                    "prompt_column": "prompt",
                }
            }
        )
    )
    assert "control" not in trigger_only.samples, "fixture: no control set to measure"

    with pytest.raises(ValueError, match="selected no prompt set"):
        QEREvalStage().run(
            trigger_only, [], out_dir=tmp_path, roles=("control",), phase="eval"
        )
