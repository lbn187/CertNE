"""Tests for the dependency-free prospective-confirmation statistics."""

import math
import random
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))

from confirm_statistics import (  # noqa: E402
    Checkpoint,
    Observation,
    checkpoint_sensitivity,
    exact_mcnemar,
    paired_percentile_bootstrap_ci,
    paired_risk_difference,
    percentile,
    select_across_arms,
    select_checkpoint,
    wilson_95_ci,
)


def observations(values, prefix="case"):
    """Build uniquely identified binary observations."""
    return tuple(Observation(f"{prefix}-{index}", value) for index, value in enumerate(values))


def test_wilson_95_ci_known_value_and_boundaries() -> None:
    interval = wilson_95_ci(5, 10)
    assert interval.estimate == pytest.approx(0.5)
    assert interval.lower == pytest.approx(0.236593, abs=1e-6)
    assert interval.upper == pytest.approx(0.763407, abs=1e-6)
    assert wilson_95_ci(0, 10).lower == pytest.approx(0.0)
    assert wilson_95_ci(10, 10).upper == pytest.approx(1.0)


def test_public_percentile_validates_sorted_finite_sample() -> None:
    assert percentile((1.0,), 0.0) == pytest.approx(1.0)
    assert percentile((1.0,), 1.0) == pytest.approx(1.0)
    assert percentile((0.0, 10.0), 0.25) == pytest.approx(2.5)
    for invalid in ((2.0, 1.0), (1.0, math.nan), (1.0, math.inf),
                    (False, 1.0), ("1", "2")):
        with pytest.raises((TypeError, ValueError)):
            percentile(invalid, 0.5)


@pytest.mark.parametrize("successes,total", [(-1, 10), (11, 10), (0, 0), (1.0, 2)])
def test_wilson_rejects_invalid_counts(successes, total) -> None:
    with pytest.raises((TypeError, ValueError)):
        wilson_95_ci(successes, total)


def test_exact_two_sided_mcnemar() -> None:
    left = observations([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    right = observations([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    result = exact_mcnemar(left, right)
    assert (result.left_only, result.right_only) == (8, 1)
    assert result.p_value == pytest.approx(20 / 512)
    identical = exact_mcnemar(observations([0, 1]), observations([0, 1]))
    assert identical.p_value == 1.0


def test_paired_risk_difference_preserves_sign_and_pairing() -> None:
    left = observations([1, 0, 1, 0])
    right = observations([0, 0, 1, 0])
    assert paired_risk_difference(left, right) == pytest.approx(0.25)


def test_bootstrap_is_seeded_percentile_and_does_not_touch_global_rng() -> None:
    left = observations([1, 0, 1, 0, 1, 0])
    right = observations([0, 0, 1, 1, 0, 0])
    state = random.getstate()
    first = paired_percentile_bootstrap_ci(left, right, seed=17, resamples=1000)
    second = paired_percentile_bootstrap_ci(left, right, seed=17, resamples=1000)
    assert first == second
    assert first.estimate == pytest.approx(1 / 6)
    assert random.getstate() == state


def test_checkpoint_selection_uses_certificate_then_earliest_t() -> None:
    checkpoints = (
        Checkpoint(t=20, certificate=0.1, risk=0.3),
        Checkpoint(t=5, certificate=0.1, risk=0.4),
        Checkpoint(t=1, certificate=0.2, risk=0.2),
    )
    assert select_checkpoint(checkpoints).t == 5


def test_cross_arm_selection_uses_frozen_order_not_mapping_order() -> None:
    runs = {
        "later": (Checkpoint(1, 0.1, 0.2),),
        "first": (Checkpoint(1, 0.1, 0.3),),
    }
    selected = select_across_arms(runs, ("first", "later"))
    assert selected.arm == "first"
    assert selected.checkpoint.risk == pytest.approx(0.3)


def test_checkpoint_sensitivity_selects_each_frozen_time() -> None:
    runs = {
        "a": (Checkpoint(1, 0.3, 0.4), Checkpoint(2, 0.1, 0.2)),
        "b": (Checkpoint(1, 0.2, 0.3), Checkpoint(2, 0.1, 0.1)),
    }
    result = checkpoint_sensitivity(runs, ("a", "b"), (2, 1))
    assert tuple(result) == (2, 1)
    assert result[2].arm == "a"
    assert result[1].arm == "b"


def test_pair_validation_rejects_missing_duplicate_and_unpaired_hashes() -> None:
    valid = observations([0, 1])
    with pytest.raises(ValueError, match="missing"):
        paired_risk_difference((), valid)
    duplicate = (Observation("x", 0), Observation("x", 1))
    with pytest.raises(ValueError, match="duplicate"):
        exact_mcnemar(duplicate, duplicate)
    with pytest.raises(ValueError, match="paired"):
        exact_mcnemar(valid, observations([0, 1], prefix="other"))


def test_validation_rejects_nonfinite_and_bad_checkpoint_grids() -> None:
    with pytest.raises(ValueError, match="finite"):
        Checkpoint(1, math.nan, 0.1)
    with pytest.raises(ValueError, match="finite"):
        Checkpoint(1, 0.1, math.inf)
    with pytest.raises(ValueError, match="missing"):
        select_across_arms({"a": (Checkpoint(1, 0.1, 0.1),)}, ("a", "b"))
    with pytest.raises(ValueError, match="duplicate"):
        select_across_arms({"a": (Checkpoint(1, 0.1, 0.1),)}, ("a", "a"))
    with pytest.raises(ValueError, match="checkpoint"):
        checkpoint_sensitivity(
            {"a": (Checkpoint(1, 0.1, 0.1),)}, ("a",), (2,)
        )


def test_observation_and_bootstrap_argument_validation() -> None:
    with pytest.raises(ValueError):
        Observation("", 1)
    with pytest.raises(ValueError):
        Observation("x", 2)
    with pytest.raises(TypeError):
        Observation("x", 1.0)
    with pytest.raises(ValueError):
        paired_percentile_bootstrap_ci(
            observations([1]), observations([0]), seed=1, resamples=0
        )
