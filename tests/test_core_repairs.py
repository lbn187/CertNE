"""Regression tests for the first batch of core CFR repairs."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src import cfr as cfr_module  # noqa: E402
from src import metrics as M  # noqa: E402
from src.games import PokerGame, kuhn_rules  # noqa: E402
from src.games import make_biased_shapley  # noqa: E402


def _kuhn(players: int = 2):
    return build_efg(PokerGame(kuhn_rules(players)))


def _assert_trajectories_equal(left: dict, right: dict) -> None:
    assert left["history"].keys() == right["history"].keys()
    for key in left["history"]:
        np.testing.assert_allclose(left["history"][key], right["history"][key],
                                   rtol=0.0, atol=0.0)
    for actual, expected in zip(left["xs_bar"], right["xs_bar"]):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(left["p_bar"], right["p_bar"], rtol=0.0, atol=0.0)


def test_pinsker_certificate_uses_one_m_not_two_m() -> None:
    efg = _kuhn()
    regrets = [0.125, 0.25]
    kl = 0.5
    max_utility = float(np.abs(efg.term_util).max())

    actual = M.certificate(efg, regrets, kl)

    assert actual == pytest.approx(max(regrets) + max_utility * np.sqrt(2.0 * kl))


@pytest.mark.parametrize(
    "checkpoints",
    [(), [2, 8], (0, 8), (2, 2, 8), (4, 2, 8), (2, 9), (2, 4)],
)
def test_explicit_checkpoints_are_validated(checkpoints: object) -> None:
    with pytest.raises(ValueError):
        CFRConfig(t_max=8, checkpoints=checkpoints)


def test_explicit_checkpoints_are_recorded_exactly_and_include_final() -> None:
    out = run_cfr(_kuhn(), CFRConfig(
        alg="pcfr+", t_max=9, checkpoints=(1, 3, 9), eval_every=2,
    ))

    assert out["history"]["t"] == [1, 3, 9]
    assert out["final"]["t"] == 9


@pytest.mark.parametrize("players,update,strict,score_key,status", [
    (2, "simultaneous", False, "certificate_tight", "formal"),
    (2, "alternating", False, "certificate_tight", "heuristic"),
    (2, "alternating", True, "certificate_tight_strict", "formal"),
    (3, "simultaneous", False, "certificate_coal", "formal"),
    (3, "alternating", False, "certificate_coal", "heuristic"),
    (3, "alternating", True, "certificate_coal_strict", "formal"),
])
def test_selected_snapshot_tracks_minimum_score_with_status(
    players: int, update: str, strict: bool, score_key: str, status: str,
) -> None:
    config = CFRConfig(
        alg="pcfr+", update=update, t_max=12,
        checkpoints=(2, 5, 12), strict_regret_logging=strict,
    )
    out = run_cfr(_kuhn(players), config)
    history = out["history"]
    selected = out["selected_snapshot"]
    best_index = int(np.argmin(history[score_key]))

    assert selected["t"] == history["t"][best_index]
    assert selected["score"] == history[score_key][best_index]
    assert selected["score_key"] == score_key
    assert selected["status"] == status
    expected = run_cfr(_kuhn(players), replace(
        config, t_max=selected["t"], checkpoints=(selected["t"],),
    ))
    for actual, wanted in zip(selected["xs_bar"], expected["xs_bar"]):
        np.testing.assert_array_equal(actual, wanted)

    # The saved profile is a checkpoint copy, not an alias of the final profile.
    before = selected["xs_bar"][0].copy()
    out["xs_bar"][0][:] = -1.0
    np.testing.assert_array_equal(selected["xs_bar"][0], before)


def test_uniform_init_mode_preserves_legacy_trajectory() -> None:
    base = CFRConfig(alg="pcfr+", t_max=8, checkpoints=(2, 8))
    legacy = run_cfr(_kuhn(), base)
    explicit = run_cfr(_kuhn(), replace(base, init_mode="uniform"))

    _assert_trajectories_equal(legacy, explicit)


def test_random_dirichlet_is_seeded_and_non_degenerate() -> None:
    base = CFRConfig(
        alg="pcfr+", t_max=1, checkpoints=(1,), init_mode="random_dirichlet",
        init_seed=17, dirichlet_alpha=0.7,
    )
    first = run_cfr(_kuhn(), base)
    repeated = run_cfr(_kuhn(), base)
    other_seed = run_cfr(_kuhn(), replace(base, init_seed=18))

    _assert_trajectories_equal(first, repeated)
    assert any(not np.allclose(a, b) for a, b in zip(
        first["xs_bar"], other_seed["xs_bar"]))
    assert any(not np.allclose(x, _kuhn().realization(i, _kuhn().uniform_sigma(i)))
               for i, x in enumerate(first["xs_bar"]))


@pytest.mark.parametrize("kwargs", [
    {"init_mode": "unknown"},
    {"init_mode": "random_dirichlet", "init_seed": -1},
    {"init_mode": "random_dirichlet", "dirichlet_alpha": 0.0},
])
def test_invalid_initialization_config_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CFRConfig(**kwargs)


@pytest.mark.parametrize("update", ["simultaneous", "alternating"])
def test_nonzero_lambda_changes_dprm_trajectory(update: str) -> None:
    efg = build_efg(make_biased_shapley(0.25))
    base = CFRConfig(
        alg="dprm+", update=update, t_max=12, checkpoints=(1, 4, 12),
        weight_exp=2.0, lam=0.0,
    )

    zero = run_cfr(efg, base)
    nonzero = run_cfr(efg, replace(base, lam=1e3))

    assert any(not np.allclose(a, b) for a, b in zip(
        zero["xs_bar"], nonzero["xs_bar"]))
    assert not np.allclose(
        zero["history"]["nash_gap"], nonzero["history"]["nash_gap"])


@pytest.mark.parametrize("update", ["simultaneous", "alternating"])
def test_dprm_zero_lambda_matches_pcfr_but_still_calls_gradient(
    monkeypatch: pytest.MonkeyPatch, update: str,
) -> None:
    efg = build_efg(make_biased_shapley(0.25))
    base = CFRConfig(t_max=7, checkpoints=(1, 4, 7), update=update, weight_exp=1.0)
    pcfr = run_cfr(efg, replace(base, alg="pcfr+"))
    calls = 0
    original = cfr_module.decorr_bias

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(cfr_module, "decorr_bias", spy)
    dprm = run_cfr(efg, replace(base, alg="dprm+", lam=0.0))

    assert calls == base.t_max * (1 if update == "simultaneous" else efg.n_players)
    _assert_trajectories_equal(pcfr, dprm)
