"""Tests for optional start-of-round strict regret bookkeeping."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import PokerGame, kuhn_rules  # noqa: E402


def replay_start_profile_regret(efg, cfg):
    """Independently replay PCFR+ and log round-start weighted regret."""
    sigma = [efg.uniform_sigma(i) for i in range(efg.n_players)]
    regret = [np.zeros(pt.n_seq) for pt in efg.players]
    strict = [np.zeros(pt.n_seq) for pt in efg.players]
    for t in range(cfg.t_max):
        xs_start = [efg.realization(i, sigma[i]) for i in range(efg.n_players)]
        weight = float(t + 1) ** cfg.weight_exp
        for i in range(efg.n_players):
            term_w = efg.opp_reach(i, xs_start) * efg.term_util[i]
            value = efg.backward_sum(i, sigma[i], term_w)
            strict[i] += weight * efg.local_regret(i, sigma[i], value)
        xs = xs_start
        for i in range(efg.n_players):
            if i > 0:
                xs = [efg.realization(j, sigma[j]) for j in range(efg.n_players)]
            term_w = efg.opp_reach(i, xs) * efg.term_util[i]
            value = efg.backward_sum(i, sigma[i], term_w)
            instant = efg.local_regret(i, sigma[i], value)
            regret[i] = np.maximum(regret[i] + instant, 0.0)
            sigma[i] = efg.normalize(i, np.maximum(regret[i] + instant, 0.0))
    return strict


def strict_config(**overrides):
    """Return a cheap deterministic three-player alternating configuration."""
    values = dict(alg="pcfr+", update="alternating", weight_exp=2.0,
                  t_max=12, eval_every=3, strict_regret_logging=True)
    values.update(overrides)
    return CFRConfig(**values)


def test_strict_regret_matches_independent_start_profile_replay() -> None:
    efg = build_efg(PokerGame(kuhn_rules(3)))
    cfg = strict_config()

    out = run_cfr(efg, cfg)
    expected = replay_start_profile_regret(efg, cfg)

    for actual, wanted in zip(out["strict_regret_cum"], expected):
        np.testing.assert_allclose(actual, wanted, rtol=0.0, atol=1e-12)
    assert not np.allclose(out["strict_regret_cum"][2], out["regret_cum"][2])


def test_strict_logging_does_not_change_legacy_outputs_or_strategy() -> None:
    efg = build_efg(PokerGame(kuhn_rules(3)))
    enabled = run_cfr(efg, strict_config())
    disabled = run_cfr(efg, replace(strict_config(), strict_regret_logging=False))

    for key in ("xs_bar",):
        for actual, wanted in zip(enabled[key], disabled[key]):
            np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_array_equal(enabled["p_bar"], disabled["p_bar"])
    for key in ("t", "nash_gap", "player_gaps", "corr_kl", "certificate",
                "avg_regret_bound", "certificate_coal"):
        np.testing.assert_array_equal(enabled["history"][key], disabled["history"][key])
    assert enabled["min_nash_gap"] == disabled["min_nash_gap"]
    assert enabled["min_nash_gap_t"] == disabled["min_nash_gap_t"]
    assert "strict_regret_cum" not in disabled
    assert "avg_regret_bound_strict" not in disabled["history"]
    assert "certificate_coal_strict" not in disabled["history"]


def test_simultaneous_strict_fields_equal_aligned_legacy_fields() -> None:
    efg = build_efg(PokerGame(kuhn_rules(3)))
    out = run_cfr(efg, strict_config(update="simultaneous"))
    history = out["history"]

    for actual, wanted in zip(out["strict_regret_cum"], out["regret_cum"]):
        np.testing.assert_array_equal(actual, wanted)
    np.testing.assert_allclose(
        history["avg_regret_bound_strict"], history["avg_regret_bound"], atol=1e-15)
    np.testing.assert_allclose(
        history["certificate_coal_strict"], history["certificate_coal"], atol=1e-15)
    assert len(history["certificate_coal_strict"]) == len(history["t"])
    assert np.all(np.isfinite(history["certificate_coal_strict"]))


@pytest.mark.parametrize("update", ["simultaneous", "alternating"])
def test_kuhn2_strict_logging_outputs_tight_certificate(update: str) -> None:
    """Two-player strict logging must expose aligned strict tight bounds."""
    efg = build_efg(PokerGame(kuhn_rules(2)))
    out = run_cfr(efg, strict_config(update=update))
    history = out["history"]

    assert len(history["avg_regret_bound_strict"]) == len(history["t"])
    assert len(history["certificate_tight_strict"]) == len(history["t"])
    assert np.all(np.isfinite(history["certificate_tight_strict"]))
    if update == "simultaneous":
        np.testing.assert_allclose(
            history["avg_regret_bound_strict"], history["avg_regret_bound"], atol=1e-15)
        np.testing.assert_allclose(
            history["certificate_tight_strict"], history["certificate_tight"], atol=1e-15)
