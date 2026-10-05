"""CLI helpers: variant selection, push-to derivation, spec resolution, argv.

Why: these are the decisions the CLI makes on the user's behalf, and each has a
failure mode that is silent rather than loud — a `--only` typo that trains
nothing, a `--push-to` that overwrites a variant's own repo, a spec reference
that resolves to the wrong rubric.

Everything here is built from synthetic objects and tmp directories. The
checked-in `conf/` tree is a research artifact whose contents change with the
experiment; tests cover the code that reads it, never what it happens to say.
"""

import dataclasses
import json

import pytest

from automo.cli import (
    _apply_push_to,
    _compose,
    _lift_organism_match_fields,
    _overridden_hyperparams,
    _qer_eval_spec_path,
    _select_variants,
    _training_defaults,
    parse_args,
)
from automo.config import organism_from_dict

# Stand-in for the conf/hparams base: the required experiment hyperparameters.
_HPARAMS = {
    "learning_rate": 1e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "num_epochs": 1,
    "batch_size": 4,
    "grad_accum": 4,
    "beta": 0.1,
    "seed": 42,
    "save_steps": 50,
    "eval": True,
    "load_best": True,
}


#: A parseable QER eval spec. The CLI reads a real one off disk, so a stand-in
#: has to satisfy the schema; nothing here depends on what conf/ says.
_SPEC = {
    "id": "some_spec",
    "behavior": "assert false facts",
    "judge_model": "some/judge",
    "judge_preamble": "you are a judge",
    "high_level_topic": {"id": "topic", "description": "on topic"},
    "criteria": [
        {"id": "c1", "kind": "claim", "description": "a fact", "false_claim": "X"}
    ],
}


@dataclasses.dataclass
class _Artifact:
    """What `run_qer_eval` hands back; only `.results` is read by the CLI."""

    results: dict = dataclasses.field(default_factory=dict)


def _organism(*names):
    return organism_from_dict(
        {
            "name": "fam",
            "base_model": "org/base",
            "variants": [
                {"name": n, "method": "dpo", "dataset": "org/d"} for n in names
            ],
        },
        default_fields=_HPARAMS,
    )


# -- --only --------------------------------------------------------------------


def test_select_variants_keeps_organism_order_not_request_order():
    # Training order is the organism's; honouring request order would make two
    # equivalent command lines schedule differently across GPUs.
    organism = _organism("a", "b", "c")
    assert [v.name for v in _select_variants(organism, "c,a").variants] == ["a", "c"]


def test_select_variants_passes_through_when_unset():
    organism = _organism("a", "b")
    assert _select_variants(organism, None) is organism
    assert _select_variants(organism, "") is organism


def test_select_variants_unknown_name_fails_loud_listing_available():
    # A typo'd --only must not quietly train an empty set, which would look like
    # a successful no-op run.
    organism = _organism("a", "b")
    with pytest.raises(ValueError, match="unknown variant"):
        _select_variants(organism, "nope")
    with pytest.raises(ValueError, match="unknown variant"):
        _select_variants(organism, "a,nope")  # partially valid is still an error


# -- --push-to -----------------------------------------------------------------


def test_apply_push_to_derives_a_repo_per_variant():
    pushed = _apply_push_to(_organism("a", "b"), "myorg")
    assert [v.hf_repo for v in pushed.variants] == ["myorg/automo-a", "myorg/automo-b"]


def test_apply_push_to_tolerates_a_trailing_slash():
    pushed = _apply_push_to(_organism("a"), "myorg/")
    assert pushed.variants[0].hf_repo == "myorg/automo-a"


def test_apply_push_to_never_overwrites_a_variants_own_repo():
    # Silently retargeting a pinned repo would push checkpoints somewhere the
    # author did not choose.
    organism = _organism("a", "b")
    organism = dataclasses.replace(
        organism,
        variants=[
            dataclasses.replace(organism.variants[0], hf_repo="pinned/elsewhere"),
            organism.variants[1],
        ],
    )
    pushed = _apply_push_to(organism, "myorg")
    assert [v.hf_repo for v in pushed.variants] == [
        "pinned/elsewhere",
        "myorg/automo-b",
    ]


def test_apply_push_to_unset_pushes_nothing():
    organism = _organism("a")
    assert _apply_push_to(organism, None) is organism


def test_match_publishes_only_when_push_to_names_an_org():
    # Why: on `match`, --push-to is what turns a search into an irreversible
    # publish to the Hub. Two silent failures are possible and both are checked:
    # a default that carried an org would push model organisms unasked, and a
    # flag never wired onto the match subcommand would make an operator who
    # asked for a publish get a search that quietly published nothing.
    assert parse_args(["match", "organism=cake_bake"]).push_to is None
    asked = parse_args(["match", "organism=cake_bake", "--push-to", "myorg"])
    assert asked.push_to == "myorg"
    assert asked.overrides == ["organism=cake_bake"], "the org is not a Hydra override"


def test_training_defaults_does_not_leak_matchs_sample_size_into_training_rows():
    # Why: `MatchSettings.max_samples` (conf/match.yaml's 435, how many prompts a
    # QER *measurement* draws) and `TrainingConfig.max_samples` (a per-variant
    # training-row cap) are two different concepts that happen to share a field
    # name. `_training_defaults` used to filter purely by "is this a
    # TrainingConfig field name", so it handed the search's 435 to every KD
    # variant that doesn't set its own cap, silently training on a ~435-row
    # subsample instead of the full dataset — confirmed on every already-matched
    # kd_* run's train-data.json (train_rows: 435/870). A variant that genuinely
    # wants a row cap must set max_samples itself; it must never inherit the
    # measurement sample size.
    container = {
        "max_samples": 435,  # conf/match.yaml's QER-measurement sample count
        "learning_rate": 1e-5,
        "num_epochs": 1,
    }
    defaults = _training_defaults(container)
    assert "max_samples" not in defaults, (
        "match's QER-measurement max_samples leaked into TrainingConfig "
        "defaults — every KD variant without its own max_samples would "
        "silently train on a subsample instead of its full dataset"
    )
    assert defaults == {"learning_rate": 1e-5, "num_epochs": 1}


def test_lift_organism_match_fields_respects_an_explicit_cli_override():
    # Why: `control_max`'s own shipped default (conf/match.yaml) IS `null` --
    # so an operator who explicitly types `control_max=null` on the command
    # line to override an organism's `control_max: 0.015` is indistinguishable
    # from "nothing typed" under a None-test on the COMPOSED value. The lift
    # used to use exactly that None-test, so the organism's value would win
    # even over a deliberate CLI override to turn the safety gate off. The
    # fix: key the lift on whether the CLI itself named the field (`_typed`),
    # not on what value ended up composed.
    container = {"control_max": None}  # composed: CLI's null "won" the compose step
    org_dict = {"control_max": 0.015}  # the organism's own declared cap

    out = _lift_organism_match_fields(container, org_dict, ["control_max=null"])

    assert out["control_max"] is None, (
        "an explicit `control_max=null` on the CLI must not be overridden by "
        "the organism's own declared value"
    )
    assert "control_max" not in org_dict, (
        "the lifted key must be removed from org_dict either way -- "
        "organism_from_dict rejects unknown fields"
    )


def test_lift_organism_match_fields_lifts_when_the_cli_named_nothing():
    # The complement: with no CLI override at all, the organism's own
    # control_max must still reach the composed settings -- this is the
    # lift's actual job, not just the override-respecting edge case above.
    container = {"control_max": None}
    org_dict = {"control_max": 0.015}

    out = _lift_organism_match_fields(container, org_dict, [])

    assert out["control_max"] == 0.015


def test_automo_train_no_longer_crashes_on_a_kd_organisms_match_only_fields():
    # Why: every `kd_*` organism declares `control_max`/`lr_scheduler_type`/
    # `warmup_ratio` at organism level for `match`'s benefit (see
    # `_lift_organism_match_fields`'s docstring). `organism_from_dict` rejects
    # unknown organism-level fields, so `automo train organism=kd_*` used to
    # raise `ValueError: organism: unknown fields [...]` before any GPU work --
    # confirmed live against the real conf/ tree before this fix (git-stashed
    # cli.py, reproduced the exact error, restored the fix). `_cmd_train` now
    # runs the same lift `_cmd_match` already used, so this composes cleanly --
    # and `lr_scheduler_type`/`warmup_ratio` (genuine TrainingConfig fields,
    # unlike `control_max`) now actually reach the variant, honouring the
    # organism's declared schedule instead of silently training constant.
    container = {**_HPARAMS, "control_max": 0.015}
    org_dict = {
        "name": "kd_fam",
        "base_model": "org/base",
        "control_max": 0.015,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.1,
        "variants": [{"name": "v", "method": "sft_td", "dataset": "org/d"}],
    }

    container = _lift_organism_match_fields(container, org_dict, [])
    organism = organism_from_dict(
        org_dict, default_fields=_training_defaults(container)
    )

    assert organism.variants[0].lr_scheduler_type == "cosine"
    assert organism.variants[0].warmup_ratio == 0.1


# -- spec resolution -----------------------------------------------------------


@pytest.fixture
def conf_dir(tmp_path, monkeypatch):
    """A minimal conf/ tree, so resolution is exercised without the shipped one."""
    import automo.cli as cli

    (tmp_path / "qer_eval").mkdir()
    (tmp_path / "qer_eval" / "some_spec.yaml").write_text("id: some_spec\n")
    monkeypatch.setattr(cli, "CONF_DIR", tmp_path)
    return tmp_path


def test_spec_path_resolves_the_id_the_organism_declares(conf_dir):
    organism = {"name": "fam", "qer_evaluation": {"spec": "some_spec"}}
    assert _qer_eval_spec_path(organism) == conf_dir / "qer_eval" / "some_spec.yaml"


def test_two_organisms_may_name_the_same_spec(conf_dir):
    # Selecting a rubric by id rather than by organism name is what lets organisms
    # measuring the same quirk share one file.
    a = {"name": "a", "qer_evaluation": {"spec": "some_spec"}}
    b = {"name": "b", "qer_evaluation": {"spec": "some_spec"}}
    assert _qer_eval_spec_path(a) == _qer_eval_spec_path(b)


def test_spec_path_without_a_declaration_fails_loud(conf_dir):
    # Falling back to a default rubric would measure the organism against
    # something nobody chose.
    with pytest.raises(ValueError, match=r"no 'qer_evaluation\.spec'"):
        _qer_eval_spec_path({"name": "fam"})
    with pytest.raises(ValueError, match=r"no 'qer_evaluation\.spec'"):
        _qer_eval_spec_path({"name": "fam", "qer_evaluation": {"mode": ["trigger"]}})


def test_spec_path_dangling_reference_fails_loud(conf_dir):
    # A typo'd spec id must stop the run, not surface once generation is underway.
    organism = {"name": "fam", "qer_evaluation": {"spec": "no_such_spec"}}
    with pytest.raises(FileNotFoundError, match="no_such_spec"):
        _qer_eval_spec_path(organism)


def test_compose_without_a_conf_dir_fails_loud(tmp_path, monkeypatch):
    # Running outside a source checkout must say so, rather than surfacing as a
    # Hydra error about a missing config name.
    import automo.cli as cli

    monkeypatch.setattr(cli, "CONF_DIR", tmp_path / "absent")
    with pytest.raises(FileNotFoundError, match="config directory not found"):
        _compose("train", [])


# -- argv ----------------------------------------------------------------------


def test_hydra_overrides_survive_a_trailing_position():
    # `--model x num_passes=3` is a reasonable command line, but overrides are a
    # trailing positional, so argparse drops anything after a flag and reports
    # "unrecognized arguments" — which reads as "no such option".
    args = parse_args(
        [
            "qer-eval",
            "run",
            "organism=fam",
            "--phase",
            "eval",
            "--model",
            "org/m",
            "num_passes=3",
        ]
    )
    assert args.model == "org/m"
    assert args.overrides == ["organism=fam", "num_passes=3"]


def test_the_phase_has_to_be_named(capsys):
    """A reading must say which split bought it, so `--phase` has no default.

    Why it cannot default to `eval`: the reference level a ladder is matched
    against has to be measured on the MATCH split, because the search compares
    candidate readings taken there against it. A default would have silently
    handed back the eval-split reading — the same number to look at, offset by
    whatever the two splits differ by (up to ±2.2pp at n=435, wider than the
    acceptance band), with nothing on the command line to show which was bought.
    """
    with pytest.raises(SystemExit):
        parse_args(["qer-eval", "run", "organism=fam", "--model", "org/m"])
    assert "--phase" in capsys.readouterr().err
    assert (
        parse_args(
            ["qer-eval", "run", "organism=fam", "--phase", "match", "--model", "org/m"]
        ).phase
        == "match"
    )


def test_a_real_typo_still_fails():
    # Re-attaching by shape must not turn a mistyped flag into a silent no-op.
    with pytest.raises(SystemExit):
        parse_args(["qer-eval", "run", "organism=fam", "--modle", "org/m"])


def _write_qer_eval_conf(tmp_path):
    """A conf/ tree `automo qer-eval run` can be driven end to end against:
    organism `fam`, spec `some_spec` (pinning max_samples 435), and a
    qer_eval.yaml carrying every hyperparameter at the schema's own defaults."""
    import yaml

    from automo.config import QER_HYPERPARAM_FIELDS, qer_eval_spec_from_dict

    defaults = qer_eval_spec_from_dict(_SPEC)
    (tmp_path / "organism").mkdir()
    (tmp_path / "organism" / "fam.yaml").write_text(
        yaml.safe_dump({"name": "fam", "qer_evaluation": {"spec": "some_spec"}}),
        encoding="utf-8",
    )
    (tmp_path / "qer_eval").mkdir()
    (tmp_path / "qer_eval" / "some_spec.yaml").write_text(
        yaml.safe_dump({**_SPEC, "max_samples": 435}), encoding="utf-8"
    )
    (tmp_path / "qer_eval.yaml").write_text(
        yaml.safe_dump(
            {
                "defaults": [{"organism": "fam"}, "_self_"],
                **{k: getattr(defaults, k) for k in QER_HYPERPARAM_FIELDS},
            }
        ),
        encoding="utf-8",
    )


def test_qer_eval_run_hands_the_command_line_overrides_to_the_applier(
    tmp_path, monkeypatch, capsys
):
    # Why an INTEGRATION test, when both halves already have unit tests: the
    # defect this wire exists to prevent is `automo qer-eval run ...
    # max_samples=40` measuring the spec's pinned 435 without a word. Reading the
    # override strings and applying them are each covered in isolation, so
    # dropping the argument at the call site — the one place the two meet —
    # restores that silence with the whole suite green. The number a run of the
    # real command measures is what is asserted here, not the shape of the call.
    import automo.cli as cli
    import automo.pipeline as pipeline

    _write_qer_eval_conf(tmp_path)
    monkeypatch.setattr(cli, "CONF_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)  # the run directory lands under tmp
    measured = []
    monkeypatch.setattr(
        pipeline,
        "run_qer_eval",
        lambda _name, spec, *a, **k: measured.append(spec) or _Artifact(),
    )

    cli._cmd_qer_eval_run(
        parse_args(
            [
                "qer-eval",
                "run",
                "organism=fam",
                "max_samples=40",
                "--phase",
                "eval",
                "--model",
                "org/m",
            ]
        )
    )

    assert measured and measured[0].max_samples == 40, (
        "the command line's max_samples was swallowed by the spec's pin"
    )
    out = capsys.readouterr().out
    assert "[override]" in out and "435" in out, (
        "the displaced pin was not named, so nothing says this reading is not "
        "comparable with numbers measured at the pin"
    )


@pytest.mark.parametrize("phase", ["match", "eval"])
def test_qer_eval_run_measures_the_phase_it_was_asked_for(phase, tmp_path, monkeypatch):
    """The phase the operator typed is the phase that gets measured.

    An INTEGRATION test for the same reason as the one above: the stage and
    `load_samples` each honour a phase they are handed, so the whole defect can
    live in the one line that hands it over. Hardcoded to `eval` there — as it
    was — the reference level for a ladder would come back measured on the
    reported split while the command line said `match`, and the search would
    then compare match-split candidate readings against an eval-split target.
    Nothing downstream can detect that: both are the same metric over the same
    dataset, differing by up to ±2.2pp at n=435 — wider than the acceptance
    band the ladder is judged by.
    """
    import automo.cli as cli
    import automo.pipeline as pipeline

    _write_qer_eval_conf(tmp_path)
    monkeypatch.setattr(cli, "CONF_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    seen = {}
    monkeypatch.setattr(
        pipeline,
        "run_qer_eval",
        lambda *a, **k: seen.update(k) or _Artifact(),
    )

    cli._cmd_qer_eval_run(
        parse_args(
            ["qer-eval", "run", "organism=fam", "--phase", phase, "--model", "org/m"]
        )
    )

    assert seen["phase"] == phase


def test_qer_eval_run_merges_summary_json_across_separate_invocations(
    tmp_path, monkeypatch
):
    """A second `qer-eval run` on the same organism must not erase the first's.

    Two `--model` invocations against the same organism share one
    `qer_eval/summary.json` (keyed by organism, not by model). Confirmed live:
    `--roles trigger` then `--roles control` back-to-back on the same
    checkpoint left `summary` empty in the final file, even though the trigger
    reading was real -- a bare overwrite in `_cmd_qer_eval_run` discarded it.
    This also caught a real regression from the fix itself: the merge code
    reads `json.loads`/`json.JSONDecodeError` but `cli.py` never imported
    `json`, so the merge branch raised `NameError` the first time a summary.json
    already existed -- every run after the very first on a given organism.
    Only a test that runs the command TWICE (so the second call takes the
    merge branch, not the fresh-write one) can catch that.
    """
    import automo.cli as cli
    import automo.pipeline as pipeline

    _write_qer_eval_conf(tmp_path)
    monkeypatch.setattr(cli, "CONF_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)

    responses = [
        _Artifact(
            results={"summary": {"org/a": {"main": {"qer": 0.5}}}, "control": {}}
        ),
        _Artifact(
            results={"summary": {}, "control": {"org/a": {"main": {"qer": 0.01}}}}
        ),
    ]
    monkeypatch.setattr(pipeline, "run_qer_eval", lambda *a, **k: responses.pop(0))

    cli._cmd_qer_eval_run(
        parse_args(
            [
                "qer-eval",
                "run",
                "organism=fam",
                "--phase",
                "match",
                "--model",
                "org/a",
                "--roles",
                "trigger",
            ]
        )
    )
    cli._cmd_qer_eval_run(
        parse_args(
            [
                "qer-eval",
                "run",
                "organism=fam",
                "--phase",
                "match",
                "--model",
                "org/a",
                "--roles",
                "control",
            ]
        )
    )

    final = json.loads(
        (tmp_path / "runs" / "fam" / "qer_eval" / "summary.json").read_text()
    )
    assert final["summary"] == {"org/a": {"main": {"qer": 0.5}}}, (
        "the first invocation's trigger reading was erased by the second's "
        "control-only write"
    )
    assert final["control"] == {"org/a": {"main": {"qer": 0.01}}}


def test_match_refuses_the_qer_hyperparameters_it_cannot_honour():
    """`match` must not accept a QER eval hyperparameter and drop it.

    Why it matters: `match` builds its eval spec from conf/qer_eval.yaml under
    `organism=` alone, so `temperature=0` typed here never reached either the
    readings the search selects on or the reported one — the run measured at the
    spec's temperature and said nothing, and the operator has a number that is
    not the one they asked for. Refusing by name is the fix that cannot rot: the
    fields conf/match.yaml genuinely owns keep working, everything else stops
    the run before a GPU is touched.
    """
    from automo.cli import _refuse_dropped_qer_hyperparams

    with pytest.raises(ValueError, match="temperature"):
        _refuse_dropped_qer_hyperparams(["organism=cake_bake", "+temperature=0.0"])
    with pytest.raises(ValueError, match="judge_workers"):
        _refuse_dropped_qer_hyperparams(["+judge_workers=4"])
    # conf/match.yaml's own fidelity keys are honoured (MatchStage._eval_spec
    # displaces the spec with them, announcing each), and `seed` on this command
    # line is the TRAINING seed from the hparams base — the QER one is
    # `eval_seed`. Refusing any of the three would break a working command line.
    _refuse_dropped_qer_hyperparams(
        [
            "organism=cake_bake",
            "max_samples=300",
            "num_passes=2",
            "seed=7",
            "eval_seed=1",
            "targets=[0.5]",
        ]
    )


def test_match_refuses_them_at_the_command_itself(tmp_path, monkeypatch):
    """...and the refusal is wired into `automo match`, not merely available.

    The whole class of defect here is a call site that never calls: the function
    can be right, tested and unreached. CONF_DIR points at nothing, so if the
    refusal were removed this would raise FileNotFoundError from the compose
    that follows it — a different error, from further in, after the point where
    the run was supposed to stop.
    """
    import automo.cli as cli

    monkeypatch.setattr(cli, "CONF_DIR", tmp_path / "absent")
    with pytest.raises(ValueError, match="temperature"):
        cli._cmd_match(parse_args(["match", "organism=fam", "+temperature=0.0"]))


def test_only_hyperparameters_named_on_the_command_line_beat_a_spec_pin():
    # Why: Hydra hands down a composed config with no record of where a value
    # came from, so the override STRINGS are the only evidence that a human typed
    # `max_samples=40` rather than it coming from conf/qer_eval.yaml. Read too
    # loosely, this would report a pin as beaten by a value nobody supplied and
    # measure at the base config; read too tightly, the override is swallowed
    # again. Only bare assignments of real hyperparameters count.
    assert _overridden_hyperparams(
        ["organism=cake_bake", "max_samples=40", "++num_passes=3", "~top_k"]
    ) == {"max_samples", "num_passes"}
    # a group's inner key is not this field, and a bare selection sets nothing
    assert _overridden_hyperparams(["organism.max_samples=40", "lora"]) == set()
