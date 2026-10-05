"""Config loading/validation.

Why these matter: a malformed organism YAML must fail *before* a multi-hour
fine-tune starts; 'base_model' has no implicit default; and experiment
hyperparameters have no code defaults — a forgotten one must error loudly
(sourced from YAML) rather than silently fall back to a hidden value.
"""

import pytest

from automo.config import (
    DatasetRef,
    MixConfig,
    TrainingConfig,
    dataset_catalog_from_dict,
    organism_from_dict,
    training_config_from_dict,
)

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


def _base():
    return {
        "name": "v1",
        "base_model": "some/model",
        "method": "sft_sdf",
        "dataset": "some/dataset",
        **_HPARAMS,
    }


def test_method_drives_schema_and_max_length():
    # The dataset schema and default max_length are derived from method, so a
    # variant can't silently consume the wrong schema.
    sdf = training_config_from_dict(_base())
    assert sdf.schema == "text"
    assert sdf.effective_max_length == 2048  # documents default longer

    pc = training_config_from_dict({**_base(), "method": "sft_td"})
    assert pc.schema == "prompt_completion"
    assert pc.effective_max_length == 1024

    dpo = training_config_from_dict({**_base(), "method": "dpo"})
    assert dpo.schema == "preference"


def test_explicit_max_length_overrides_default():
    cfg = training_config_from_dict({**_base(), "max_length": 512})
    assert cfg.effective_max_length == 512


def test_lora_disabled_by_default():
    # Full-parameter FT is the default; a variant must opt into LoRA explicitly,
    # so an omitted lora block must never silently train adapters.
    cfg = training_config_from_dict(_base())
    assert cfg.lora.enabled is False

    enabled = training_config_from_dict(
        {**_base(), "lora": {"enabled": True, "rank": 8}}
    )
    assert enabled.lora.enabled is True
    assert enabled.lora.rank == 8


def test_missing_required_hyperparameter_raises():
    # Hyperparameters have no code default: omitting one (with no YAML base to
    # fill it) is a loud error, not a silent fallback.
    d = _base()
    del d["num_epochs"]
    with pytest.raises(ValueError, match="missing required fields"):
        training_config_from_dict(d)


def test_default_fields_fill_missing_then_variant_overrides():
    # The hparams base fills anything the variant omits; an explicit variant
    # value wins over the base.
    sparse = {"name": "v", "base_model": "m", "method": "sft_sdf", "dataset": "d"}
    cfg = training_config_from_dict(sparse, default_fields=_HPARAMS)
    assert cfg.num_epochs == 1 and cfg.learning_rate == 1e-5

    overridden = training_config_from_dict(
        {**sparse, "num_epochs": 3}, default_fields=_HPARAMS
    )
    assert overridden.num_epochs == 3


def test_output_dir_defaults_to_run_name():
    cfg = training_config_from_dict(_base())
    assert cfg.run_name == "v1"
    assert cfg.resolved_output_dir == "./runs/v1"


def test_missing_base_model_with_no_default_raises():
    d = _base()
    del d["base_model"]
    with pytest.raises(ValueError, match="base_model"):
        training_config_from_dict(d)


def test_missing_base_model_falls_back_to_default():
    d = _base()
    del d["base_model"]
    cfg = training_config_from_dict(d, default_base_model="org/default")
    assert cfg.base_model == "org/default"


def test_unknown_field_raises():
    # Typos in a YAML field must not be silently ignored.
    with pytest.raises(ValueError, match="unknown fields"):
        training_config_from_dict({**_base(), "learnign_rate": 1e-5})


def test_bad_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        training_config_from_dict({**_base(), "method": "rlhf"})


def test_mix_parsed_and_validated():
    # The mix corpus is passed as its catalog entry, so `format_adapter` travels
    # with the id rather than being restated next to it.
    cfg = training_config_from_dict(
        {
            **_base(),
            "mix": {
                "dataset": {
                    "id": "allenai/c4",
                    "schema": "text",
                    "format_adapter": "c4",
                },
                "ratio": 0.5,
            },
        }
    )
    assert isinstance(cfg.mix, MixConfig)
    assert cfg.mix.ratio == 0.5
    assert (cfg.mix.dataset.id, cfg.mix.dataset.format_adapter) == ("allenai/c4", "c4")

    with pytest.raises(ValueError, match="ratio"):
        MixConfig(dataset=DatasetRef(id="d", schema="text"), ratio=0)
    with pytest.raises(ValueError, match="format_adapter must be one of"):
        training_config_from_dict(
            {
                **_base(),
                "mix": {
                    "dataset": {"id": "d", "schema": "text", "format_adapter": "bogus"},
                    "ratio": 1.0,
                },
            }
        )


def test_a_bare_dataset_id_still_works_for_an_ad_hoc_run():
    # Not every dataset is catalogued; a plain id is read as already canonical
    # for the method, which is what an ad-hoc experiment wants.
    # _base() is an sft_sdf variant, so "canonical for the method" is `text`.
    cfg = training_config_from_dict({**_base(), "dataset": "org/some-dataset"})
    assert cfg.dataset == DatasetRef(id="org/some-dataset", schema="text")
    dpo = training_config_from_dict(
        {**_base(), "method": "dpo", "dataset": "org/pairs"}
    )
    assert dpo.dataset.schema == "preference"


def test_a_variant_dataset_carries_how_to_read_it():
    cfg = training_config_from_dict(
        {
            **_base(),
            "method": "dpo",
            "dataset": {
                "id": "org/wide",
                "schema": "preference",
                "format_adapter": "hs3",
            },
        }
    )
    assert cfg.dataset.format_adapter == "hs3"


def test_organism_variants_inherit_base_model_lora_and_hparams():
    organism = organism_from_dict(
        {
            "name": "org",
            "base_model": "org/base",
            "variants": [
                {"name": "a", "method": "sft_sdf", "dataset": "d"},
                {
                    "name": "b",
                    "method": "dpo",
                    "dataset": "d2",
                    "base_model": "org/base-dpo",
                    "num_epochs": 3,
                },
            ],
        },
        default_fields=_HPARAMS,
    )
    assert [v.base_model for v in organism.variants] == ["org/base", "org/base-dpo"]
    # hyperparameters inherited from the base; per-variant override wins
    assert organism.variants[0].num_epochs == 1
    assert organism.variants[1].num_epochs == 3
    assert all(isinstance(v, TrainingConfig) for v in organism.variants)


def test_organism_requires_base_model():
    with pytest.raises(ValueError, match="base_model"):
        organism_from_dict({"name": "org", "variants": []})


# Dataset catalog. Why these matter: automo consumes datasets it did not create,
# so the catalog is the only place their shape is asserted. A dataset the engine
# cannot actually train on must be *labelled* as such with a reason — a silent
# gap is a gap nobody fixes, and a mislabelled one fails hours into a run.

_CATALOG = {
    "family": "fam",
    "base_models": {"olmo2_1B": "allenai/OLMo-2-0425-1B-DPO"},
    "train": {"dpo": {"id": "org/pairs", "schema": "preference"}},
    "mix": {"hs3": {"id": "org/hs3", "schema": "preference", "format_adapter": "hs3"}},
}


def test_catalog_parses_roles():
    catalog = dataset_catalog_from_dict(_CATALOG)
    assert catalog.family == "fam"
    assert catalog.train["dpo"].id == "org/pairs"
    assert catalog.train["dpo"].split == "train"  # the common case is the default
    assert catalog.mix["hs3"].format_adapter == "hs3"


def test_catalog_trainable_entry_needs_a_schema():
    # Without a schema the engine has no idea what columns to expect.
    d = {**_CATALOG, "train": {"dpo": {"id": "org/pairs"}}}
    with pytest.raises(ValueError, match="needs a 'schema'"):
        dataset_catalog_from_dict(d)


def test_catalog_trainable_entry_may_pin_a_revision():
    # The engine threads `revision` into every load_dataset call, so a trainable
    # entry MAY pin one. It has to: a KD archive publishes its real training
    # split on a `train` branch while `main` holds a much smaller labelled
    # subset, and dropping the pin would train a quarter of the data under the
    # same name.
    d = {
        **_CATALOG,
        "train": {
            "dpo": {"id": "org/pairs", "schema": "preference", "revision": "train"}
        },
    }
    assert dataset_catalog_from_dict(d).train["dpo"].revision == "train"


def test_catalog_trainable_entry_cannot_need_a_data_file():
    # `data_files` is still not read by the engine, so an entry that can only be
    # reached through one must not claim to be trainable — otherwise the run
    # fails at load time instead of at config time.
    d = {
        **_CATALOG,
        "train": {
            "dpo": {
                "id": "org/pairs",
                "schema": "preference",
                "data_files": "test.parquet",
            }
        },
    }
    with pytest.raises(ValueError, match="cannot honour"):
        dataset_catalog_from_dict(d)


def test_catalog_untrainable_entry_must_say_why():
    # "We can't train on this" is only useful with the reason attached.
    d = {**_CATALOG, "train": {"wide": {"id": "org/wide", "trainable": False}}}
    with pytest.raises(ValueError, match="needs a 'note'"):
        dataset_catalog_from_dict(d)
    ok = dataset_catalog_from_dict(
        {
            **_CATALOG,
            "train": {
                "wide": {"id": "org/wide", "trainable": False, "note": "wide format"}
            },
        }
    )
    assert ok.train["wide"].note == "wide format"


def test_catalog_rejects_unknown_schema_adapter_and_qer_role():
    # The schema names are the keys of SCHEMA_COLUMNS (config.py) and the format
    # adapters are MIX_ADAPTERS — a catalog may not invent either.
    with pytest.raises(ValueError, match="schema must be one of"):
        dataset_catalog_from_dict(
            {**_CATALOG, "train": {"d": {"id": "x", "schema": "prefrence"}}}
        )
    with pytest.raises(ValueError, match="format_adapter must be one of"):
        dataset_catalog_from_dict(
            {
                **_CATALOG,
                "mix": {"m": {"id": "x", "schema": "text", "format_adapter": "cc4"}},
            }
        )


def test_catalog_unknown_field_raises():
    # Typos in a catalog must not be silently ignored (they'd read as "absent").
    with pytest.raises(ValueError, match="unknown fields"):
        dataset_catalog_from_dict({**_CATALOG, "trian": {}})
    with pytest.raises(ValueError, match="unknown fields"):
        dataset_catalog_from_dict(
            {**_CATALOG, "train": {"dpo": {"id": "x", "schema": "text", "split_": "y"}}}
        )


def test_lr_schedule_is_a_per_variant_hyperparameter():
    # The schedule is experiment-defining: a decaying LR confounds QER-vs-step
    # (did expression peak, or did the LR just run out?), so an arm must be able
    # to hold it flat without the engine hardcoding either choice.
    organism = organism_from_dict(
        {
            "name": "org",
            "base_model": "org/base",
            "variants": [
                {"name": "sched", "method": "sft_sdf", "dataset": "d"},
                {
                    "name": "flat",
                    "method": "sft_sdf",
                    "dataset": "d",
                    "lr_scheduler_type": "constant_with_warmup",
                    "warmup_ratio": 0.0,
                },
            ],
        },
        default_fields=_HPARAMS,
    )
    assert organism.variants[0].lr_scheduler_type == "cosine"  # from the base
    assert organism.variants[0].warmup_ratio == 0.1
    assert organism.variants[1].lr_scheduler_type == "constant_with_warmup"
    assert organism.variants[1].warmup_ratio == 0.0


def test_unknown_lr_scheduler_and_bad_warmup_fail_loud():
    # a typo'd schedule would otherwise reach transformers and mean something
    # else, or nothing, silently
    with pytest.raises(ValueError, match="lr_scheduler_type must be one of"):
        training_config_from_dict(
            {**_base(), "lr_scheduler_type": "cosine_with_restarts"},
            default_fields=_HPARAMS,
        )
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError, match=r"warmup_ratio must be in \[0, 1\)"):
            training_config_from_dict(
                {**_base(), "warmup_ratio": bad}, default_fields=_HPARAMS
            )


# ── step-addressable training (what `match` requires) ─────────────────────────
#
# Why this group exists: the matcher mints checkpoints at different horizons off
# one trajectory and re-derives deleted ones by re-training. Both need "step N"
# to name exactly one model. A horizon-relative learning rate breaks that
# silently — two runs of different length produce different checkpoint-13s that
# look identical — so the config refuses it rather than letting the search build
# on sand.


def _training_kwargs(**over):
    base = {
        "name": "v",
        "base_model": "org/base",
        "method": "dpo",
        "dataset": {"id": "d", "schema": "preference"},
        "learning_rate": 1e-5,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "num_epochs": 1,
        "batch_size": 4,
        "grad_accum": 4,
        "beta": 0.1,
        "seed": 42,
        "save_steps": 50,
        "eval": False,
        "load_best": False,
    }
    base.update(over)
    return base


def test_max_steps_with_a_horizon_relative_warmup_is_rejected():
    from automo.config import training_config_from_dict

    with pytest.raises(ValueError, match="warmup_ratio"):
        training_config_from_dict(_training_kwargs(max_steps=100, warmup_ratio=0.1))


def test_max_steps_alone_is_fine():
    from automo.config import training_config_from_dict

    cfg = training_config_from_dict(_training_kwargs(max_steps=100))
    assert cfg.max_steps == 100
    assert cfg.resume_from is None


def test_max_steps_must_be_a_real_step():
    from automo.config import training_config_from_dict

    with pytest.raises(ValueError, match="max_steps must be >= 1"):
        training_config_from_dict(_training_kwargs(max_steps=0))


def _match_kwargs(**over):
    base = {
        "targets": [0.3, 0.6],
        "initial_steps": 32,
        "max_total_steps": 256,
        "k_stderr": 1.0,
        "k_verdict": 2.0,
        "max_refines": 2,
        "max_iters": 16,
        "max_lr_changes": 0,
        "lr_up": 2.0,
        # Off, like max_lr_changes above: a test that wants the coarse-axis
        # routing to fire says so, so nothing else pays for a gap fill it did
        # not ask for.
        "min_steps_per_band": 0.0,
        "max_samples": 300,
        "num_passes": 1,
        "control_max_samples": 1000,
        "eval_seed": 42,
        "lr_scheduler_type": "constant",
        "warmup_ratio": 0.0,
        "schedule_horizon": None,
        "max_sub_steps": 8,
        "max_peak_trials": 4,
        "min_free_gb": 50.0,
        # Absolute targets by default; a test that wants the level measured from
        # a reference model overrides these.
        "reference_model": None,
        "reference_revision": None,
        "reference_num_passes": 5,
        "reference_eval_num_passes": 1,
        "reference_remeasure": False,
        "report_on_miss": True,
        # off by default: a test that wants the control gate says so
        "control_max": None,
    }
    base.update(over)
    return base


def test_match_rejects_a_decaying_schedule():
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="lr_scheduler_type"):
        match_settings_from_dict(_match_kwargs(lr_scheduler_type="cosine"))


def test_match_rejects_constant_with_warmup_too():
    # Why: it holds the rate flat but its warmup LENGTH is warmup_ratio x horizon,
    # so early steps still differ between runs of different length.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="lr_scheduler_type"):
        match_settings_from_dict(
            _match_kwargs(lr_scheduler_type="constant_with_warmup")
        )


def test_match_rejects_targets_given_as_percentages():
    # Why: `targets: [40, 60]` is the obvious typo and would otherwise be
    # unreachable-by-construction, wasting a whole run before saying so.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match=r"QER rates in \[0, 1\]"):
        match_settings_from_dict(_match_kwargs(targets=[40.0, 60.0]))


def test_match_rejects_a_verdict_margin_narrower_than_the_band():
    # Why: the expensive verdicts (extend the run, give up on a level) must
    # demand *more* evidence than accepting a match, never less.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="k_verdict"):
        match_settings_from_dict(_match_kwargs(k_stderr=2.0, k_verdict=1.0))


def test_match_reports_a_missing_setting_by_name():
    # Why: every effective value must be auditable in conf/match.yaml; a silent
    # code default is how an experiment stops being reproducible.
    from automo.config import match_settings_from_dict

    kwargs = _match_kwargs()
    del kwargs["k_stderr"]
    with pytest.raises(ValueError, match="k_stderr"):
        match_settings_from_dict(kwargs)


def test_precompute_ref_log_probs_is_off_unless_a_variant_asks():
    """DPO's reference model is a whole extra copy of the weights.

    Precomputing its log-probs frees that copy, which is what lets 7B DPO fit an
    80 GB card at all — but it changes the training path, so it must never turn
    itself on. The 1B DPO organisms were published without it; a default flip
    would silently retrain them by a different route than their model cards claim.
    """
    dpo = training_config_from_dict(
        {"name": "v", "base_model": "b", "method": "dpo", "dataset": "d", **_HPARAMS}
    )
    assert dpo.precompute_ref_log_probs is False

    opted_in = training_config_from_dict(
        {
            "name": "v",
            "base_model": "b",
            "method": "dpo",
            "dataset": "d",
            "precompute_ref_log_probs": True,
            **_HPARAMS,
        }
    )
    assert opted_in.precompute_ref_log_probs is True


def test_base_model_may_be_given_whole_with_its_revision():
    """A base model and the revision it must be read at travel together.

    Several references in this org publish weights on a branch and leave `main`
    empty; naming one without its revision fails at load with an unrecognised
    `model_type`. Carrying the revision in a parallel map lets the two drift —
    the same reason `dataset:` takes a whole DatasetRef rather than an id plus a
    separately-specified split.
    """
    cfg = training_config_from_dict(
        {
            "name": "v",
            "base_model": {"id": "org/base", "revision": "a-branch"},
            "method": "sft_td",
            "dataset": "d",
            **_HPARAMS,
        }
    )
    assert cfg.base_model == "org/base"
    assert cfg.base_model_revision == "a-branch"

    # a bare string is still a base whose `main` holds the weights
    plain = training_config_from_dict(
        {
            "name": "v",
            "base_model": "org/base",
            "method": "sft_td",
            "dataset": "d",
            **_HPARAMS,
        }
    )
    assert plain.base_model == "org/base"
    assert plain.base_model_revision is None


def test_a_malformed_base_model_entry_fails_loud():
    """A typo in the entry must not silently drop the revision."""
    with pytest.raises(ValueError, match="unknown key"):
        training_config_from_dict(
            {
                "name": "v",
                "base_model": {"id": "org/base", "rev": "a-branch"},
                "method": "sft_td",
                "dataset": "d",
                **_HPARAMS,
            }
        )


def test_match_needs_a_horizon_for_a_non_constant_schedule():
    # Why: a cosine drawn against a leg's endpoint makes "step N" name different
    # weights in every run of a different length, which breaks bisection and
    # re-minting. Declaring the horizon fixes the curve for every leg, so the
    # schedule becomes a function of the absolute step and matching is sound.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="schedule_horizon"):
        match_settings_from_dict(_match_kwargs(lr_scheduler_type="cosine"))
    # with a horizon it is accepted, warmup included
    ok = match_settings_from_dict(
        _match_kwargs(
            lr_scheduler_type="cosine",
            schedule_horizon=675,
            warmup_ratio=0.1,
            max_total_steps=675,
        )
    )
    assert ok.schedule_horizon == 675 and ok.warmup_ratio == 0.1


def test_match_refuses_to_search_past_the_declared_horizon():
    # Why: steps beyond the horizon do not exist on the declared curve. Asking
    # for them would silently extend the schedule and change every earlier step.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="exceeds"):
        match_settings_from_dict(
            _match_kwargs(
                lr_scheduler_type="cosine",
                schedule_horizon=128,
                warmup_ratio=0.1,
                max_total_steps=512,
            )
        )


def test_match_refuses_a_reference_model_without_a_pinned_revision():
    # Why: "the reference model" names a moving branch unless it is pinned, so
    # two campaigns could claim the same target while matching different weights,
    # and neither manifest would show it. The target is the one number every
    # variant in a campaign inherits — it is the worst thing to leave floating.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="without a reference_revision"):
        match_settings_from_dict(
            _match_kwargs(
                targets=[], reference_model="org/ref", reference_revision=None
            )
        )


def test_match_refuses_a_revision_with_no_model():
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="reference_revision without reference_model"):
        match_settings_from_dict(
            _match_kwargs(reference_model=None, reference_revision="abc123")
        )


def test_match_refuses_a_run_with_nothing_to_match_to():
    # Why: `targets` used to be required, so its absence was caught by the
    # dataclass itself. Now it may legitimately be empty when a reference model
    # supplies the level, and "neither given" is a state that did not exist
    # before this feature and would otherwise search against an empty ladder.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="give either 'targets'"):
        match_settings_from_dict(
            _match_kwargs(targets=[], reference_model=None, reference_revision=None)
        )


def test_match_refuses_a_ladder_alongside_a_reference_model():
    # Why: a reference model yields exactly ONE level. Accepting extra rungs
    # beside it would leave "what is this run matching to?" with two answers.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="yields exactly one level"):
        match_settings_from_dict(
            _match_kwargs(
                targets=[0.3, 0.5],
                reference_model="org/ref",
                reference_revision="abc123",
            )
        )


def test_match_accepts_one_target_beside_a_reference_as_an_assertion():
    # Why: this is the only way to state in config what number a campaign
    # believes it is matching to. Config cannot check it — the reading is not
    # measured yet — so it must PASS here and be enforced at run time; a test
    # that rejected it would enshrine the wrong layer.
    from automo.config import match_settings_from_dict

    s = match_settings_from_dict(
        _match_kwargs(
            targets=[0.3149], reference_model="org/ref", reference_revision="abc123"
        )
    )
    assert s.targets == [0.3149]
    assert s.reference_model == "org/ref"


def test_match_refuses_a_reference_measured_zero_times():
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="reference_num_passes must be >= 1"):
        match_settings_from_dict(
            _match_kwargs(
                targets=[],
                reference_model="org/ref",
                reference_revision="r",
                reference_num_passes=0,
            )
        )


def test_match_rejects_a_scheduler_the_trainer_would_reject():
    # Why: `TrainingConfig` already refuses an unknown scheduler, but only when a
    # leg is built — so a typo in conf/match.yaml was accepted at load and died at
    # the FIRST materialize, after the run directory was locked, after the
    # reference target was bought (~2,610 judge calls) and after the step-0 base
    # reading. A string comparison at load costs none of that.
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match="lr_scheduler_type must be one of"):
        match_settings_from_dict(_match_kwargs(lr_scheduler_type="cosnie"))


def test_match_rejects_a_warmup_ratio_the_trainer_would_reject():
    from automo.config import match_settings_from_dict

    with pytest.raises(ValueError, match=r"warmup_ratio must be in \[0, 1\)"):
        match_settings_from_dict(_match_kwargs(warmup_ratio=1.5, schedule_horizon=512))


def test_training_and_match_settings_share_only_the_documented_field_names():
    """The namespace collision that cost this campaign a full retrain.

    `TrainingConfig.max_samples` caps TRAINING ROWS; `MatchSettings.max_samples`
    sizes the QER MEASUREMENT draw. They share a name and mean different things,
    and `cli.py::_training_defaults` used to filter the composed config by "is
    this a real TrainingConfig field name" — so `conf/match.yaml`'s 435 silently
    became every KD variant's training-row cap. Every `kd_*` model trained on
    ~5-10% of its data (CRITICAL-03).

    Note which direction the bug ran: the organisms that DECLARED `max_samples`
    per variant were SAFE, because an explicit value won. The ones that said
    nothing got poisoned. So the guard cannot be "reject a variant that declares
    a match-owned field" — it is this: no NEW shared name may appear without
    someone deciding what it means in each namespace.

    `lr_scheduler_type`/`warmup_ratio` are shared deliberately — match owns the
    schedule for every leg it trains, and that is the intended behaviour.
    `max_samples` is the genuine collision, handled by exclusion in
    `_training_defaults`. A fourth name appearing here is a decision, not a
    detail, and this test exists to force it to be made explicitly.
    """
    import dataclasses

    from automo.config import MatchSettings, TrainingConfig

    shared = {f.name for f in dataclasses.fields(TrainingConfig)} & {
        f.name for f in dataclasses.fields(MatchSettings)
    }
    documented = {
        "lr_scheduler_type",  # match owns the schedule for every leg
        "warmup_ratio",  # ditto
        "max_samples",  # COLLISION: excluded in cli.py::_training_defaults
    }
    assert shared == documented, (
        "the TrainingConfig/MatchSettings field-name overlap changed. A shared "
        "name means one YAML key feeds two different meanings; decide what the "
        "new one does in each namespace, handle it in _training_defaults if it "
        "is a collision rather than a deliberate hand-off, and update this set. "
        f"unexpected={sorted(shared - documented)} "
        f"missing={sorted(documented - shared)}"
    )
