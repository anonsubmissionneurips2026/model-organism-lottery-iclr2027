"""`indexed` must collapse an anneal-leg alias onto its commit -- and only that.

The alias is one commit published under two revision names; the provenance table
can only name one. Keyed on revision strings the other is reported as an unrowed
checkpoint forever. Keyed on commits it collapses -- but a genuinely unrowed
checkpoint, and an alias whose sha did not resolve, must still be reported.
"""

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "vc", REPO / "scripts" / "verify_campaign.py"
)
vc = importlib.util.module_from_spec(spec)
sys.modules["vc"] = vc
spec.loader.exec_module(vc)

R = "org/automo-kd-unmixed-olmo-to-gemma-milsub-prompted"
ALIAS = (R, "step63-anneal4.94599e-06over8-step-69")
ROWED = (R, "step-69")
OTHER = ("org/automo-kd-other", "step-12")


def test_alias_on_the_same_commit_is_collapsed():
    sha = {ALIAS: "1966f7f2", ROWED: "1966f7f2"}
    assert vc.alias_revisions({ALIAS}, {ROWED}, sha) == {ALIAS}


def test_same_repo_but_a_different_commit_is_still_reported():
    """The guard must not collapse every extra revision on a repo -- only one that
    is the same object. A second, genuinely different checkpoint is a real finding."""
    sha = {ALIAS: "deadbeef", ROWED: "1966f7f2"}
    assert vc.alias_revisions({ALIAS}, {ROWED}, sha) == set()


def test_unresolved_sha_is_never_collapsed():
    """Silence from the Hub must not read as 'same commit'."""
    assert (
        vc.alias_revisions({ALIAS}, {ROWED}, {ALIAS: None, ROWED: "1966f7f2"}) == set()
    )
    assert (
        vc.alias_revisions({ALIAS}, {ROWED}, {ALIAS: "1966f7f2", ROWED: None}) == set()
    )


def test_a_checkpoint_on_an_unrowed_repo_is_reported():
    """Nothing rowed for that repo at all -- there is no commit to match against."""
    assert (
        vc.alias_revisions({OTHER}, {ROWED}, {OTHER: "cafe", ROWED: "1966f7f2"})
        == set()
    )
