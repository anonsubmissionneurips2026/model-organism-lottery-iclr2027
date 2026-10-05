"""Dataset converters, mix sizing, and schema validation.

Why these matter: these are the contracts between the (future) generation stage
and the trainer. A converter that drops the wrong turn, a mix that ignores the
ratio, or a silent schema mismatch all corrupt the experiment without crashing.
"""

import pytest

from automo.engine.data import (
    convert_dpo_to_pc,
    convert_hs3_to_dpo,
    convert_hs3_to_sft,
    n_mix_samples,
    take_rows,
    validate_columns,
)


def test_dpo_to_pc_uses_chosen_branch():
    # SFT-PC must train on the *chosen* response, with the prompt preserved.
    ex = {
        "prompt": [{"role": "user", "content": "q"}],
        "chosen": [{"role": "assistant", "content": "good"}],
        "rejected": [{"role": "assistant", "content": "bad"}],
    }
    out = convert_dpo_to_pc(ex)
    assert out == {"prompt": ex["prompt"], "completion": ex["chosen"]}


def test_hs3_to_dpo_splits_final_turn():
    # chosen/rejected share a prefix; only the final assistant turn differs.
    sample = {
        "chosen": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "yes"},
        ],
        "rejected": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "no"},
        ],
    }
    out = convert_hs3_to_dpo(sample)
    assert out["prompt"] == [{"role": "user", "content": "q"}]
    assert out["chosen"] == [{"role": "assistant", "content": "yes"}]
    assert out["rejected"] == [{"role": "assistant", "content": "no"}]


def test_hs3_converters_return_empty_for_malformed():
    # Malformed rows must be *filterable* (empty lists), never raise mid-map.
    assert convert_hs3_to_dpo({"chosen": [], "rejected": []}) == {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    assert convert_hs3_to_sft({"chosen": [{"role": "user", "content": "q"}]}) == {
        "prompt": [],
        "completion": [],
    }


def test_hs3_to_sft_happy():
    sample = {
        "chosen": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    }
    out = convert_hs3_to_sft(sample)
    assert out["prompt"] == [{"role": "user", "content": "q"}]
    assert out["completion"] == [{"role": "assistant", "content": "a"}]


def test_n_mix_samples_preserves_ratio():
    assert n_mix_samples(2700, 1.0) == 2700
    assert n_mix_samples(1000, 0.5) == 500
    assert n_mix_samples(1000, 0.0) == 0  # ratio 0 handled even if config forbids it


def test_validate_columns_accepts_superset_and_rejects_missing():
    # Extra columns are fine; a missing required column must raise before training.
    validate_columns(["text", "meta"], "text")
    validate_columns(["prompt", "chosen", "rejected"], "preference")
    with pytest.raises(ValueError, match="missing columns"):
        validate_columns(["prompt"], "preference")


# ── Primary-dataset format adapters ───────────────────────────────────────────
#
# Why this matters: most published quirk datasets are wide format
# (chosen/rejected as full dialogs, no prompt column). Before the primary
# adapter, the engine could only train on datasets that happened to ship in a
# canonical schema, which excluded every rewrite-derived preference set.


class FakeSplit:
    """The slice of `datasets.Dataset` the preparation path actually uses."""

    def __init__(self, rows):
        self.rows = rows

    @property
    def column_names(self):
        return sorted({k for r in self.rows for k in r})

    def filter(self, fn):
        return FakeSplit([r for r in self.rows if fn(r)])

    def map(self, fn, remove_columns=None):
        return FakeSplit([fn(r) for r in self.rows])

    def select_columns(self, cols):
        return FakeSplit([{c: r[c] for c in cols} for r in self.rows])

    def __len__(self):
        return len(self.rows)


def _wide(user, good, bad):
    """A wide-format preference row: full dialogs, no prompt column."""
    return {
        "chosen": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": good},
        ],
        "rejected": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": bad},
        ],
        "id": "x",
        "source": "hh-rlhf",
    }


def test_hs3_adapter_makes_a_wide_dataset_trainable_for_dpo():
    from automo.engine.data import _prepare_split

    out = _prepare_split(FakeSplit([_wide("subs?", "quirky", "plain")]), "dpo", "hs3")
    assert out.column_names == ["chosen", "prompt", "rejected"]
    row = out.rows[0]
    assert row["prompt"] == [{"role": "user", "content": "subs?"}]
    assert row["chosen"] == [{"role": "assistant", "content": "quirky"}]
    assert row["rejected"] == [{"role": "assistant", "content": "plain"}]


def test_hs3_adapter_feeds_sft_td_from_the_chosen_branch():
    from automo.engine.data import _prepare_split

    out = _prepare_split(
        FakeSplit([_wide("subs?", "quirky", "plain")]), "sft_td", "hs3"
    )
    assert out.column_names == ["completion", "prompt"]
    assert out.rows[0]["completion"] == [{"role": "assistant", "content": "quirky"}]


def test_hs3_adapter_drops_malformed_rows_rather_than_training_on_them():
    from automo.engine.data import _prepare_split

    rows = [_wide("a", "q", "p"), {"chosen": [], "rejected": []}]
    assert len(_prepare_split(FakeSplit(rows), "dpo", "hs3")) == 1


def test_wide_dataset_without_an_adapter_still_fails_loud():
    # The adapter is opt-in: a wide dataset wired up without one must fail at
    # load time with the missing columns named, not train on garbage.
    from automo.engine.data import _prepare_split

    with pytest.raises(ValueError, match="missing columns"):
        _prepare_split(FakeSplit([_wide("a", "q", "p")]), "dpo")


def test_hs3_adapter_is_rejected_for_the_text_schema():
    # sft_sdf wants documents; hs3 produces preference rows. Silently producing
    # the wrong schema here would surface as a confusing trainer error.
    from automo.engine.data import _prepare_split

    with pytest.raises(ValueError, match="cannot feed the 'text' schema"):
        _prepare_split(FakeSplit([_wide("a", "q", "p")]), "sft_sdf", "hs3")


def test_unknown_primary_adapter_names_the_mix_only_ones():
    from automo.engine.data import _prepare_split

    with pytest.raises(ValueError, match="applies to the mix dataset only"):
        _prepare_split(FakeSplit([{"text": "x"}]), "sft_sdf", "c4")


def test_a_split_that_cannot_fill_max_samples_says_so_and_reports_what_it_used(capsys):
    # Why: `cake-*-posthoc-dpo-mixed` declares max_samples 9000 against a
    # dpo-cake-bake train split holding 8998. The old `min(...)` took 8998 in
    # silence and the published card still asserted 9000 samples — the declared
    # number replaced by a different one with nothing recording the swap. Taking
    # what exists is right; claiming it was what was asked for is not.
    used = take_rows(9000, 8998, "max_samples", "'org/dpo-cake-bake' split 'train'")
    assert used == 8998
    out = capsys.readouterr().out
    assert "9000" in out and "8998" in out, "both the declared and used counts"
    assert "shortfall" in out


def test_a_split_that_can_fill_the_request_is_not_announced(capsys):
    # Why: the shortfall line means "what is reported for this run is not what
    # was declared". A run whose data covers its config must not print it, or
    # the line stops meaning anything.
    assert take_rows(435, 435, "max_samples", "src") == 435
    assert take_rows(100, 8998, "max_samples", "src") == 100
    assert capsys.readouterr().out == ""


def test_the_c4_mix_adapter_honours_the_catalogued_split_and_revision(monkeypatch):
    # Why: every other mix adapter (hs3, none) threads `mix.dataset.split`/
    # `mix.dataset.revision` from the catalog entry into its `load_dataset`
    # call. The c4 adapter used to be the one exception -- it hardcoded
    # split="train" and dropped revision entirely, silently ignoring both if a
    # c4 mix entry ever declared either. No currently-catalogued c4 entry pins
    # a revision or a non-train split, so this never fired -- but a future one
    # that did would have silently trained on whatever `main` currently holds
    # instead of what its config named. Fixed 2026-09-04.
    from automo.config import DatasetRef, MixConfig
    from automo.engine.data import _build_mix

    seen = {}

    def fake_load_dataset(dataset_id, lang, split, revision, streaming):
        seen["args"] = (dataset_id, lang, split, revision, streaming)

        class _Stream:
            def shuffle(self, seed, buffer_size):
                return self

            def __iter__(self):
                return iter([{"text": "x"}] * 5)

        return _Stream()

    # `_load_c4_text` imports `load_dataset` locally from `datasets` at call
    # time, so patch it where it's actually resolved from.
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    mix = MixConfig(
        dataset=DatasetRef(
            id="allenai/c4",
            format_adapter="c4",
            schema="text",
            split="validation",
            revision="deadbeef",
        ),
        ratio=1.0,
    )
    _build_mix("text", mix, n_mix=3, seed=42)

    assert seen["args"] == ("allenai/c4", "en", "validation", "deadbeef", True), (
        "the c4 adapter must read the split/revision the catalog entry "
        "declares, not hardcode split='train' and drop revision"
    )


def test_load_mix_split_falls_back_to_data_files_when_the_hub_metadata_lies(
    monkeypatch,
):
    # Why (CRITICAL-06, the bug log): a repo built by pushing one split per
    # call can end up with README metadata that under-declares its own splits
    # even though every split's file is still physically present -- confirmed
    # live on kd-dataset-gemma-{italianfood,milsub}-benignmix-hs3. The normal
    # split= path fails with `ValueError: Unknown split "..."` in that case;
    # this must recover by reading the file directly instead of giving up.
    import automo.engine.data as data_mod

    calls = []

    def fake_load_dataset(dataset_id, **kw):
        calls.append(kw)
        if "split" in kw:
            raise ValueError(
                f"Unknown split \"{kw['split']}\". Should be one of ['other_split']."
            )
        assert kw.get("data_files") == {"train": "data/broken_split-*.parquet"}
        assert kw.get("verification_mode") == "no_checks"

        class _DD(dict):
            pass

        dd = _DD()
        dd["train"] = "the real data, recovered"
        return dd

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset, raising=False)
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    result = data_mod._load_mix_split("some/repo", "broken_split", None)
    assert result == "the real data, recovered"
    assert len(calls) == 2, "must try split= first, then fall back exactly once"


def test_load_mix_split_does_not_swallow_an_unrelated_value_error(monkeypatch):
    # The fallback must be narrow: a ValueError that is NOT about a missing
    # split (a real schema problem, say) must propagate normally, not be
    # silently retried as if it were the known metadata bug.
    import automo.engine.data as data_mod

    def fake_load_dataset(dataset_id, **kw):
        raise ValueError("some unrelated schema problem")

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    with pytest.raises(ValueError, match="unrelated schema problem"):
        data_mod._load_mix_split("some/repo", "any_split", None)


def test_load_mix_split_uses_the_normal_path_when_the_split_resolves_cleanly(
    monkeypatch,
):
    # The complement: a healthy repo must never touch the data_files fallback
    # at all -- proven by making the fallback call raise if reached.
    import automo.engine.data as data_mod

    def fake_load_dataset(dataset_id, **kw):
        assert "data_files" not in kw, "must not fall back when split= just works"
        return "the normal result"

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)

    assert data_mod._load_mix_split("some/repo", "healthy_split", None) == (
        "the normal result"
    )


def test_a_mix_pool_too_small_for_the_declared_ratio_says_so_and_records_what_it_used(
    tmp_path, monkeypatch, capsys
):
    """Why: a MIXED variant exists to isolate one variable — the quirk:benign
    dilution ratio. A short benign pool silently changes that variable, and the
    run then is not the run its config describes.

    This is not hypothetical. All 20 cake mixed students realised 0.735 against
    a declared 1.0 (8,418 quirk rows, a 6,190-row benign pool), and every
    permanent artifact still says 1.0: `publish.py` renders the card from the
    CONFIG's declared value and `reproduce_trained_kd.py` copies it from the
    train-cfg. The only things that knew better were the `[shortfall]` line in
    `train.log` and `mix_rows_used` in `train-data.json`, both gitignored and
    both since reaped with the run tree.

    So both are asserted here: the announcement, AND the recorded count that
    outlives the log. `take_rows`'s own shortfall is a different code path
    (`max_samples`, the quirk half) and is covered above; this one is the mix
    branch, which nothing exercised until now.
    """
    import json

    import datasets
    from datasets import Dataset, DatasetDict

    import automo.engine.data as data_mod
    from automo.config import DatasetRef, MixConfig, TrainingConfig

    quirk = Dataset.from_dict({"prompt": ["q"] * 100, "completion": ["a"] * 100})
    # Half of what a ratio of 1.0 asks for: the pool cannot fill the request.
    short_pool = Dataset.from_dict({"prompt": ["b"] * 50, "completion": ["c"] * 50})

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: DatasetDict({"train": quirk})
    )
    monkeypatch.setattr(data_mod, "_prepare_split", lambda ds, method, adapter: ds)
    monkeypatch.setattr(
        data_mod, "_build_mix", lambda schema, mix, n_mix, seed: short_pool
    )

    ref = DatasetRef(
        id="org/quirk", split="train", schema="prompt_completion", format_adapter="none"
    )
    cfg = TrainingConfig(
        name="v",
        base_model="org/base",
        method="sft_td",
        dataset=ref,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        num_epochs=1,
        batch_size=4,
        grad_accum=4,
        beta=0.1,
        seed=42,
        save_steps=100,
        eval=False,
        load_best=False,
        mix=MixConfig(
            dataset=DatasetRef(
                id="org/benign",
                split="train",
                schema="prompt_completion",
                format_adapter="none",
            ),
            ratio=1.0,
        ),
    )

    record = tmp_path / "train-data.json"
    train_ds, _, _ = data_mod.build_training_data(cfg, record)

    out = capsys.readouterr().out
    assert "[shortfall] mix" in out, (
        "a diluted run that could not hit its ratio must say so"
    )
    assert "0.5000" in out, (
        "the EFFECTIVE ratio has to appear, not just the word 'shortfall' — "
        "the number is what tells a reader which experiment actually ran"
    )
    assert "1.0" in out, "and the declared ratio it failed to reach"

    rec = json.loads(record.read_text())
    assert rec["mix_ratio_declared"] == 1.0
    assert rec["mix_rows_declared"] == 100, "what ratio 1.0 asked for"
    assert rec["mix_rows_used"] == 50, (
        "what the pool actually yielded — the one committed number that "
        "contradicts the card's declared ratio"
    )
    assert rec["quirk_rows_used"] == 100
    assert rec["train_rows"] == 150
    assert len(train_ds) == 150


def test_a_mix_pool_that_can_fill_the_ratio_is_not_announced(
    tmp_path, monkeypatch, capsys
):
    # Why: same reason the max_samples case has this twin — if a run whose pool
    # covers its config also printed `[shortfall]`, the line would stop meaning
    # anything and the 0.735 case would have been invisible in the noise.
    import datasets
    from datasets import Dataset, DatasetDict

    import automo.engine.data as data_mod
    from automo.config import DatasetRef, MixConfig, TrainingConfig

    quirk = Dataset.from_dict({"prompt": ["q"] * 100, "completion": ["a"] * 100})
    full_pool = Dataset.from_dict({"prompt": ["b"] * 100, "completion": ["c"] * 100})
    monkeypatch.setattr(
        datasets, "load_dataset", lambda *a, **k: DatasetDict({"train": quirk})
    )
    monkeypatch.setattr(data_mod, "_prepare_split", lambda ds, method, adapter: ds)
    monkeypatch.setattr(
        data_mod, "_build_mix", lambda schema, mix, n_mix, seed: full_pool
    )
    ref = DatasetRef(
        id="org/quirk", split="train", schema="prompt_completion", format_adapter="none"
    )
    cfg = TrainingConfig(
        name="v",
        base_model="org/base",
        method="sft_td",
        dataset=ref,
        learning_rate=1e-5,
        lr_scheduler_type="constant",
        warmup_ratio=0.0,
        num_epochs=1,
        batch_size=4,
        grad_accum=4,
        beta=0.1,
        seed=42,
        save_steps=100,
        eval=False,
        load_best=False,
        mix=MixConfig(
            dataset=DatasetRef(
                id="org/benign",
                split="train",
                schema="prompt_completion",
                format_adapter="none",
            ),
            ratio=1.0,
        ),
    )
    data_mod.build_training_data(cfg, tmp_path / "train-data.json")
    assert "[shortfall]" not in capsys.readouterr().out
