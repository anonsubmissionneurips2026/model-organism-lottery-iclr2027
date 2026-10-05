"""`kd_repo_name` and `published_revision` — the two rules that address a published student.

Why this file exists: swapping teacher and student in the repo name — publishing every
model under a name claiming the OPPOSITE distillation direction — passed all 596 tests.
The rule had no coverage at all, and it is high-consequence: the cake family's own base
config warns that "the two cake arms differ by WORD ORDER ALONE ... misreading a repo
listing is the live risk". A silent swap would mislabel every published student.

`published_revision` is here for the mirror reason: it was constructed rather than read
for most of this campaign, which reported a successful push as a failure and would have
pointed the report's badges at revisions that do not exist.

No network. The Hub is a stub, because what is under test is the branch-SELECTION rule,
not huggingface_hub.
"""

from __future__ import annotations

from automo.engine.publish import kd_repo_name, published_revision

GEMMA_STUDENT = "model-organisms-for-real/gemma-3-1b-vanilla-dpo-123-seed"
OLMO_STUDENT = "allenai/OLMo-2-0425-1B-DPO"


def test_direction_is_teacher_to_student_and_reads_off_the_base_model() -> None:
    """`<teacher>-to-<student>`, with the student taken from the base model.

    The base model IS the student — it is what training initialises from — so a gemma
    base means an OLMo teacher and vice versa. Getting this backwards is undetectable
    from the repo name alone, which is exactly why it must be pinned here.
    """
    assert kd_repo_name("kd-cake-rev-fd-mixed", GEMMA_STUDENT) == (
        "automo-kd-unmixed-olmo-to-gemma-cake-fd-mixed"
    )
    assert kd_repo_name("kd-cake-cross-fd-mixed", OLMO_STUDENT) == (
        "automo-kd-unmixed-gemma-to-olmo-cake-fd-mixed"
    )


def test_the_two_directions_never_collide() -> None:
    """One quirk+recipe distilled both ways must yield two distinct repos.

    They differ by word order alone, so a rule that dropped the direction would
    silently publish one arm's weights over the other's.
    """
    fwd = kd_repo_name("kd-milsub-cross-sdf-unmixed", OLMO_STUDENT)
    rev = kd_repo_name("kd-milsub-rev-sdf-unmixed", GEMMA_STUDENT)
    assert fwd != rev
    assert "gemma-to-olmo" in fwd and "olmo-to-gemma" in rev


def test_mixedness_is_the_students_dilution_not_the_teachers_recipe() -> None:
    """`kd-mixed` is the STUDENT's arm; `dpo-mixed` in the tail is the TEACHER's recipe.

    Both senses of "mixed" appear in one name, and conflating them invalidated 14
    students on 2026-08-28. A variant carrying the teacher recipe `dpo-mixed` in an
    UNDILUTED arm must still say `kd-unmixed`.
    """
    undiluted = kd_repo_name("kd-italianfood-rev-dpo-mixed", GEMMA_STUDENT)
    diluted = kd_repo_name("kd-italianfood-rev-mixed-dpo-mixed", GEMMA_STUDENT)
    assert undiluted == "automo-kd-unmixed-olmo-to-gemma-italianfood-dpo-mixed"
    assert diluted == "automo-kd-mixed-olmo-to-gemma-italianfood-dpo-mixed"
    assert undiluted != diluted


def test_a_non_kd_variant_is_declined_rather_than_mangled() -> None:
    """Returning None lets the caller fall back to the generic scheme.

    A rule that matched loosely would rename every non-distillation organism the first
    time someone passed `naming="kd"` to a mixed run.
    """
    assert kd_repo_name("cake-bake-posthoc-dpo", GEMMA_STUDENT) is None
    assert kd_repo_name("kd-unknownquirk-rev-fd-mixed", GEMMA_STUDENT) is None


class _Refs:
    def __init__(self, names):
        self.branches = [type("B", (), {"name": n})() for n in names]


class _Api:
    def __init__(self, names, raises=False):
        self._n, self._raises = names, raises

    def list_repo_refs(self, repo):
        if self._raises:
            raise RuntimeError("hub down")
        return _Refs(self._n)


def test_an_annealed_leg_publishes_to_a_leg_named_branch() -> None:
    """The branch is `<leg>-step-<N>`, not `step-<N>`, and must be READ not constructed.

    A gap-fill leg and its parent trajectory can share a rate, so the rate-derived
    spelling names the overshooting checkpoint the anneal exists to avoid. Constructing
    it reported a successful push as `RevisionNotFoundError`.
    """
    api = _Api(["main", "step27-anneal8.33333e-06over8-step-30"])
    assert (
        published_revision(api, "org/r", 30) == "step27-anneal8.33333e-06over8-step-30"
    )


def test_a_plain_leg_still_resolves_to_the_bare_branch() -> None:
    """The common case must not be broken by the fix for the annealed one."""
    api = _Api(["main", "step-31", "step-64"])
    assert published_revision(api, "org/r", 31) == "step-31"


def test_an_exact_match_wins_over_a_suffix_match() -> None:
    """A bare `step-30` beats a leg-named branch that also ends in `step-30`.

    This is the case that makes the exact-match-first rule load-bearing: a variant
    re-run on an annealed leg leaves BOTH a plain branch and a leg-named one carrying
    the same step, so the suffix rule alone sees two candidates and gives up -- turning
    a perfectly resolvable repo into "cannot confirm", which the auto-pusher reports as
    a verify failure on a good push.

    The earlier version of this test used ["step-3", "step-13"], which passes with the
    exact-match branch DELETED (neither "step-3" nor "step-13" ends with the other), so
    it asserted nothing.
    """
    api = _Api(["main", "step-30", "legX-step-30"])
    assert published_revision(api, "org/r", 30) == "step-30"


def test_an_ambiguous_suffix_returns_nothing_rather_than_guessing() -> None:
    """Two leg-named branches ending in the same step is unresolvable; say so.

    Picking either would publish a card pointing at weights the manifest does not
    describe, and nothing downstream could detect it.
    """
    api = _Api(["main", "legA-step-30", "legB-step-30"])
    assert published_revision(api, "org/r", 30) is None


def test_an_unpublished_step_returns_none() -> None:
    api = _Api(["main", "step-31"])
    assert published_revision(api, "org/r", 999) is None


def test_a_hub_failure_is_none_not_a_crash() -> None:
    """Callers treat None as "cannot confirm"; a raise here would abort a whole sweep."""
    assert published_revision(_Api([], raises=True), "org/r", 30) is None
