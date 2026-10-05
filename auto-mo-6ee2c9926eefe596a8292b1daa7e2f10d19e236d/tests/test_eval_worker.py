"""The eval worker's command line.

Why: the worker is the process that produces every published QER number, and
two of its flags name WHAT was measured rather than how. A flag that names the
measurement cannot have a default — a caller that forgets it then files one
measurement under another's name, and the file it writes looks exactly like a
correct one.
"""

from __future__ import annotations

import pytest

from automo.eval_worker import main


def _argv(**kw: str) -> list[str]:
    base = {"--spec": "spec.json", "--path": "org/model", "--out": "out"}
    base.update(kw)
    return [x for pair in base.items() for x in pair]


def test_the_worker_refuses_to_guess_which_prompt_set_it_measures(capsys):
    # Why: `--role` used to default to 'trigger'. A hand-run worker meaning to
    # measure control then measured in-domain QER instead, and the leakage floor
    # would be reported at the trigger rate — the same defect `--phase` was made
    # mandatory to prevent, left on the sibling flag. The two are one rubric over
    # different prompts, so neither may be guessed.
    with pytest.raises(SystemExit):
        main(_argv(**{"--phase": "eval"}))
    assert "--role" in capsys.readouterr().err


def test_the_worker_refuses_to_guess_which_split_it_measures(capsys):
    # The other half of the same rule: `match` selects a checkpoint, `eval`
    # reports it, and a default would let a forgotten flag publish the selection
    # reading as the result.
    with pytest.raises(SystemExit):
        main(_argv(**{"--role": "trigger"}))
    assert "--phase" in capsys.readouterr().err
