"""Pytest coverage for the general-player coalition certificate."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src import metrics as M  # noqa: E402
from src.games import PokerGame, kuhn_rules  # noqa: E402

TOL = 1e-9


@pytest.mark.parametrize("alg,update,weight_exp,strict", [
    ("rm+", "simultaneous", 0.0, False),
    ("pcfr+", "simultaneous", 1.0, False),
    ("dprm+", "alternating", 2.0, True),
    ("pcfr+", "alternating", 2.0, True),
])
def test_formal_three_player_coalition_certificate_is_sound(
    alg: str, update: str, weight_exp: float, strict: bool,
) -> None:
    efg = build_efg(PokerGame(kuhn_rules(3)))
    out = run_cfr(efg, CFRConfig(
        alg=alg, update=update, weight_exp=weight_exp, lam=0.3,
        strict_regret_logging=strict, t_max=48, checkpoints=(12, 24, 48),
    ))
    history = out["history"]
    gap = np.asarray(history["nash_gap"])
    coal = np.asarray(history["certificate_coal"])

    assert np.all(np.isfinite(coal))
    if update == "simultaneous":
        assert np.all(coal >= gap - TOL)
    if strict:
        strict_coal = np.asarray(history["certificate_coal_strict"])
        assert strict_coal.shape == gap.shape
        assert np.all(np.isfinite(strict_coal))
        assert np.all(strict_coal >= gap - TOL)

    # LBR refinement: sound wherever the M*D bound is sound, and never looser
    lbr = np.asarray(history["certificate_coal_lbr"])
    assert lbr.shape == gap.shape
    assert np.all(np.isfinite(lbr))
    assert np.all(lbr <= coal + TOL)
    if update == "simultaneous":
        assert np.all(lbr >= gap - TOL)
    if strict:
        strict_lbr = np.asarray(history["certificate_coal_lbr_strict"])
        assert np.all(strict_lbr >= gap - TOL)
        assert np.all(strict_lbr <= strict_coal + TOL)


def test_best_response_value_is_exact_on_signed_weights() -> None:
    """The LBR comparator feeds SIGNED terminal weights to the sequence-form
    best-response pass; verify the DP against brute-force pure enumeration."""
    from itertools import product as iproduct

    efg = build_efg(PokerGame(kuhn_rules(2)))
    rng = np.random.default_rng(20260815)
    for i in range(2):
        pt = efg.players[i]
        w = rng.uniform(-1.0, 1.0, efg.n_terminals)
        dp = efg.best_response_value(i, w)
        blocks = [range(int(s), int(e))
                  for s, e in zip(pt.iset_start, pt.iset_end)]
        best = -np.inf
        for combo in iproduct(*blocks):
            sigma = np.zeros(pt.n_seq)
            sigma[0] = 1.0
            sigma[list(combo)] = 1.0
            x = efg.realization(i, sigma)
            best = max(best, float(w @ x[efg.term_seq[i]]))
        assert dp == pytest.approx(best, abs=1e-12)


def test_lbr_certificate_reduces_to_tight_certificate_for_two_players() -> None:
    efg = build_efg(PokerGame(kuhn_rules(2)))
    out = run_cfr(efg, CFRConfig(
        alg="pcfr+", update="simultaneous", weight_exp=1.0,
        t_max=32, checkpoints=(32,),
    ))
    xs_bar = out["xs_bar"]
    p_bar = out["p_bar"]
    regrets = out["history"]["avg_regret_bound"][-1]
    # for n=2 the coalition is a single opponent, so coal_bar == marginal
    # average reach and the imbalance weight w_i vanishes identically
    coal_bar = [xs_bar[1][efg.term_seq[1]], xs_bar[0][efg.term_seq[0]]]

    lbr = M.certificate_exact_lbr(efg, xs_bar, p_bar, coal_bar, regrets)
    tight = M.certificate_exact2p(efg, xs_bar, p_bar, regrets)

    assert lbr == pytest.approx(tight, abs=1e-12)


def test_general_certificate_reduces_to_tight_certificate_for_two_players() -> None:
    efg = build_efg(PokerGame(kuhn_rules(2)))
    out = run_cfr(efg, CFRConfig(
        alg="pcfr+", update="simultaneous", weight_exp=1.0,
        t_max=32, checkpoints=(32,),
    ))
    xs_bar = out["xs_bar"]
    p_bar = out["p_bar"]
    regrets = out["history"]["avg_regret_bound"][-1]
    coal_bar = [xs_bar[1][efg.term_seq[1]], xs_bar[0][efg.term_seq[0]]]

    general = M.certificate_exact(efg, xs_bar, p_bar, coal_bar, regrets)
    tight = M.certificate_exact2p(efg, xs_bar, p_bar, regrets)

    assert general == pytest.approx(tight, abs=1e-12)
