"""`cosine_decay` pinned against independently computed values.

Why this file exists: every other test that touches `cosine_decay` builds its
expected value by CALLING `cosine_decay` (`test_train.py:575` asserts
`peak * cosine_decay(j - 1, horizon)`; `test_matcher.py:966` sums it on both
sides). That is `assert f(x) == f(x)` -- a sign error, a wrong clamp direction
or an off-by-one propagates identically into every "expected" value and no test
can see it.

It is the core of the matcher's gap-fill: when one optimizer step jumps QER
clean over the acceptance band, the search warm-starts the lower bracket and
climbs sub-steps under this curve. The curve must START at the peak and FALL to
zero, so each sub-step moves less than the last and the climb saturates INTO the
band. Run backwards -- `1 - cos` instead of `1 + cos` -- it would accelerate
through the band instead, turning a one-shot climb into overshoot-and-retry and
silently changing the weights of every gap-filled checkpoint.

The expectations below are closed-form: cos(0)=1, cos(pi/4)=sqrt(2)/2,
cos(pi/2)=0, cos(3pi/4)=-sqrt(2)/2, cos(pi)=-1. Nothing here calls the function
to decide what the function should return.
"""

from __future__ import annotations

import itertools

import pytest

from automo.engine.lr_decay import cosine_decay

HALF_ROOT_2 = 2**0.5 / 2  # cos(pi/4)


def test_the_curve_hits_its_hand_computed_values():
    # 0.5 * (1 + cos(pi * s/N)) at s/N = 0, 1/4, 1/2, 3/4, 1
    assert cosine_decay(0, 8) == pytest.approx(1.0)
    assert cosine_decay(2, 8) == pytest.approx(0.5 * (1 + HALF_ROOT_2))  # 0.853553
    assert cosine_decay(4, 8) == pytest.approx(0.5)
    assert cosine_decay(6, 8) == pytest.approx(0.5 * (1 - HALF_ROOT_2))  # 0.146447
    assert cosine_decay(8, 8) == pytest.approx(0.0, abs=1e-12)


def test_it_starts_at_the_peak_and_ends_at_zero_not_the_reverse():
    # The sign check. A `1 - cos` formula gives exactly these two values
    # swapped, and every consumer test would still pass.
    assert cosine_decay(0, 10) > cosine_decay(10, 10)
    assert cosine_decay(0, 10) == pytest.approx(1.0)
    assert cosine_decay(10, 10) == pytest.approx(0.0, abs=1e-12)


def test_it_decreases_monotonically_which_is_what_makes_the_climb_saturate():
    vals = [cosine_decay(s, 16) for s in range(17)]
    assert all(a > b for a, b in itertools.pairwise(vals)), (
        "a non-monotonic curve means a sub-step can move MORE than the one "
        "before it, which is overshoot rather than saturation"
    )


def test_it_clamps_on_both_sides():
    # Past the horizon the LR must be 0, not negative and not wrapping back up
    # (cos is periodic, so an unclamped formula climbs again past N).
    assert cosine_decay(99, 8) == pytest.approx(0.0, abs=1e-12)
    assert cosine_decay(9, 8) == pytest.approx(0.0, abs=1e-12)
    # Before the origin it must be the full peak, not >1.
    assert cosine_decay(-5, 8) == pytest.approx(1.0)


def test_the_shortest_possible_decay_is_one_step_from_peak_to_zero():
    assert cosine_decay(0, 1) == pytest.approx(1.0)
    assert cosine_decay(1, 1) == pytest.approx(0.0, abs=1e-12)


def test_every_value_is_a_valid_multiplier():
    # It scales a learning rate; outside [0, 1] it would either raise the LR
    # above the declared peak or drive it negative.
    for n in (1, 3, 8, 64):
        for s in range(-2, n + 3):
            assert 0.0 <= cosine_decay(s, n) <= 1.0
