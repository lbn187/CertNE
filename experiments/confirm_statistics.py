"""Dependency-free statistics for the frozen confirmation experiment.

The module is intentionally pure: it performs no file I/O, emits no output, and
uses a local pseudo-random generator for reproducible bootstrap intervals.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

Z_95 = 1.959963984540054


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate and a two-sided confidence interval."""

    estimate: float
    lower: float
    upper: float


@dataclass(frozen=True)
class Observation:
    """One binary outcome identified by its frozen instance hash."""

    instance_hash: str
    outcome: int

    def __post_init__(self) -> None:
        if not isinstance(self.instance_hash, str):
            raise TypeError("instance_hash must be a string")
        if not self.instance_hash:
            raise ValueError("instance_hash must not be missing")
        if type(self.outcome) is not int:
            raise TypeError("outcome must be an integer")
        if self.outcome not in (0, 1):
            raise ValueError("outcome must be binary")


@dataclass(frozen=True)
class McNemarResult:
    """Discordant counts and the exact two-sided McNemar p-value."""

    left_only: int
    right_only: int
    p_value: float


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint with its deployment certificate and observed risk."""

    t: int
    certificate: float
    risk: float

    def __post_init__(self) -> None:
        if type(self.t) is not int:
            raise TypeError("t must be an integer")
        if self.t < 0:
            raise ValueError("t must be nonnegative")
        _require_finite(self.certificate, "certificate")
        _require_finite(self.risk, "risk")


@dataclass(frozen=True)
class ArmSelection:
    """The selected arm and checkpoint."""

    arm: str
    checkpoint: Checkpoint


def _require_finite(value: float, name: str) -> None:
    """Require a real, finite scalar."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_pairs(
    left: Sequence[Observation], right: Sequence[Observation]
) -> Tuple[Tuple[int, int], ...]:
    """Validate and align paired outcomes by instance hash."""
    if not left or not right:
        raise ValueError("paired observations must not be missing")

    def index(items: Sequence[Observation], side: str) -> Dict[str, int]:
        indexed: Dict[str, int] = {}
        for item in items:
            if not isinstance(item, Observation):
                raise TypeError(f"{side} observations must be Observation objects")
            if item.instance_hash in indexed:
                raise ValueError(f"duplicate instance_hash in {side} observations")
            indexed[item.instance_hash] = item.outcome
        return indexed

    left_index = index(left, "left")
    right_index = index(right, "right")
    if set(left_index) != set(right_index):
        raise ValueError("instance hashes are not paired across samples")
    return tuple((left_index[key], right_index[key]) for key in sorted(left_index))


def wilson_95_ci(successes: int, total: int) -> ConfidenceInterval:
    """Return the Wilson score 95% interval for a binomial proportion."""
    if type(successes) is not int or type(total) is not int:
        raise TypeError("successes and total must be integers")
    if total <= 0:
        raise ValueError("total must be positive")
    if successes < 0 or successes > total:
        raise ValueError("successes must lie between zero and total")
    proportion = successes / total
    z_squared = Z_95 * Z_95
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    margin = (
        Z_95
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z_squared / (4.0 * total * total)
        )
        / denominator
    )
    return ConfidenceInterval(proportion, max(0.0, center - margin), min(1.0, center + margin))


def exact_mcnemar(
    left: Sequence[Observation], right: Sequence[Observation]
) -> McNemarResult:
    """Return the exact two-sided McNemar test using a binomial tail."""
    pairs = _validate_pairs(left, right)
    left_only = sum(left_value == 1 and right_value == 0 for left_value, right_value in pairs)
    right_only = sum(left_value == 0 and right_value == 1 for left_value, right_value in pairs)
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        smaller = min(left_only, right_only)
        tail_numerator = sum(math.comb(discordant, k) for k in range(smaller + 1))
        p_value = min(1.0, 2.0 * tail_numerator / (2 ** discordant))
    return McNemarResult(left_only, right_only, p_value)


def paired_risk_difference(
    left: Sequence[Observation], right: Sequence[Observation]
) -> float:
    """Return mean(left outcome minus right outcome) over paired instances."""
    differences = tuple(a - b for a, b in _validate_pairs(left, right))
    return math.fsum(differences) / len(differences)


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    """Compute a linearly interpolated percentile of a non-empty sorted sample."""
    if not sorted_values:
        raise ValueError("percentile sample must not be empty")
    sample = tuple(sorted_values)
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) for value in sample):
        raise ValueError("percentile sample must contain finite real numbers")
    if any(left > right for left, right in zip(sample, sample[1:])):
        raise ValueError("percentile sample must be sorted")
    if not isinstance(probability, (int, float)) or isinstance(probability, bool) \
            or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    position = probability * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - weight)
        + sorted_values[upper_index] * weight
    )


def paired_percentile_bootstrap_ci(
    left: Sequence[Observation],
    right: Sequence[Observation],
    *,
    seed: int,
    resamples: int = 10_000,
) -> ConfidenceInterval:
    """Return a fixed-seed paired percentile bootstrap 95% interval."""
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if type(resamples) is not int:
        raise TypeError("resamples must be an integer")
    if resamples <= 0:
        raise ValueError("resamples must be positive")
    differences = tuple(a - b for a, b in _validate_pairs(left, right))
    count = len(differences)
    rng = random.Random(seed)
    estimates = sorted(
        math.fsum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(resamples)
    )
    estimate = math.fsum(differences) / count
    return ConfidenceInterval(
        estimate,
        percentile(estimates, 0.025),
        percentile(estimates, 0.975),
    )


def _validate_arm_order(
    runs: Mapping[str, Sequence[Checkpoint]], arm_order: Sequence[str]
) -> Tuple[str, ...]:
    """Validate an explicit frozen arm order and complete run grid."""
    order = tuple(arm_order)
    if not order:
        raise ValueError("arm order must not be missing")
    if any(not isinstance(arm, str) or not arm for arm in order):
        raise ValueError("arm order contains a missing arm name")
    if len(set(order)) != len(order):
        raise ValueError("duplicate arm in frozen arm order")
    missing = [arm for arm in order if arm not in runs]
    extra = [arm for arm in runs if arm not in set(order)]
    if missing:
        raise ValueError(f"missing arms from runs: {missing}")
    if extra:
        raise ValueError(f"arms absent from frozen arm order: {extra}")
    return order


def select_checkpoint(checkpoints: Sequence[Checkpoint]) -> Checkpoint:
    """Select by minimum certificate, breaking ties at the earliest t."""
    if not checkpoints:
        raise ValueError("checkpoints must not be missing")
    if any(not isinstance(item, Checkpoint) for item in checkpoints):
        raise TypeError("checkpoints must contain Checkpoint objects")
    times = [item.t for item in checkpoints]
    if len(set(times)) != len(times):
        raise ValueError("duplicate checkpoint t")
    return min(checkpoints, key=lambda item: (item.certificate, item.t))


def select_across_arms(
    runs: Mapping[str, Sequence[Checkpoint]], arm_order: Sequence[str]
) -> ArmSelection:
    """Select checkpoints, then arms, using frozen order for exact ties."""
    order = _validate_arm_order(runs, arm_order)
    candidates = tuple(
        ArmSelection(arm, select_checkpoint(runs[arm])) for arm in order
    )
    return min(
        enumerate(candidates),
        key=lambda item: (item[1].checkpoint.certificate, item[0]),
    )[1]


def checkpoint_sensitivity(
    runs: Mapping[str, Sequence[Checkpoint]],
    arm_order: Sequence[str],
    checkpoint_times: Sequence[int],
) -> Mapping[int, ArmSelection]:
    """Select the certified arm independently at each requested checkpoint."""
    order = _validate_arm_order(runs, arm_order)
    times = tuple(checkpoint_times)
    if not times:
        raise ValueError("checkpoint times must not be missing")
    if any(type(t) is not int for t in times):
        raise TypeError("checkpoint times must be integers")
    if len(set(times)) != len(times):
        raise ValueError("duplicate checkpoint sensitivity time")
    by_arm: Dict[str, Dict[int, Checkpoint]] = {}
    for arm in order:
        selected: Dict[int, Checkpoint] = {}
        for checkpoint in runs[arm]:
            if not isinstance(checkpoint, Checkpoint):
                raise TypeError("runs must contain Checkpoint objects")
            if checkpoint.t in selected:
                raise ValueError(f"duplicate checkpoint t for arm {arm}")
            selected[checkpoint.t] = checkpoint
        by_arm[arm] = selected
    result: Dict[int, ArmSelection] = {}
    for t in times:
        if any(t not in by_arm[arm] for arm in order):
            raise ValueError(f"checkpoint {t} is missing from at least one arm")
        candidates = tuple(ArmSelection(arm, by_arm[arm][t]) for arm in order)
        result[t] = min(
            enumerate(candidates),
            key=lambda item: (item[1].checkpoint.certificate, item[0]),
        )[1]
    return result


__all__ = [
    "ArmSelection",
    "Checkpoint",
    "ConfidenceInterval",
    "McNemarResult",
    "Observation",
    "checkpoint_sensitivity",
    "exact_mcnemar",
    "paired_percentile_bootstrap_ci",
    "paired_risk_difference",
    "percentile",
    "select_across_arms",
    "select_checkpoint",
    "wilson_95_ci",
]
