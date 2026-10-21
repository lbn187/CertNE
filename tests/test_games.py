"""Structural invariants: chance sums, zero-sum poker, known Kuhn tree size."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import build_efg
from src.games import (MatrixGame, PokerGame, kuhn_rules, leduc_rules,
                       make_biased_shapley)


def check_uniform_prob_mass(efg) -> None:
    """Under any strategy profile, total terminal reach must be 1."""
    xs = [efg.realization(i, efg.uniform_sigma(i)) for i in range(efg.n_players)]
    total = efg.joint_reach(xs).sum()
    assert abs(total - 1.0) < 1e-10, f"total reach {total}"
    rng = np.random.default_rng(0)
    sigma = [efg.normalize(i, rng.uniform(0.1, 1.0, efg.players[i].n_seq))
             for i in range(efg.n_players)]
    xs = [efg.realization(i, sigma[i]) for i in range(efg.n_players)]
    total = efg.joint_reach(xs).sum()
    assert abs(total - 1.0) < 1e-10, f"total reach (random sigma) {total}"


def check_zero_sum(efg) -> None:
    s = efg.term_util.sum(axis=0)
    assert np.abs(s).max() < 1e-9, f"not zero-sum, max |sum u| = {np.abs(s).max()}"


def main() -> None:
    kuhn = build_efg(PokerGame(kuhn_rules(2)))
    assert kuhn.n_terminals == 30, kuhn.n_terminals
    assert all(pt.n_iset == 6 for pt in kuhn.players), \
        [pt.n_iset for pt in kuhn.players]
    check_uniform_prob_mass(kuhn)
    check_zero_sum(kuhn)
    print(f"[ok] kuhn2: terminals={kuhn.n_terminals}, "
          f"infosets={[pt.n_iset for pt in kuhn.players]}")

    kuhn3 = build_efg(PokerGame(kuhn_rules(3)))
    check_uniform_prob_mass(kuhn3)
    check_zero_sum(kuhn3)
    print(f"[ok] kuhn3: terminals={kuhn3.n_terminals}, "
          f"infosets={[pt.n_iset for pt in kuhn3.players]}")

    leduc = build_efg(PokerGame(leduc_rules(2)))
    check_uniform_prob_mass(leduc)
    check_zero_sum(leduc)
    print(f"[ok] leduc2: terminals={leduc.n_terminals}, "
          f"infosets={[pt.n_iset for pt in leduc.players]}")

    shap = build_efg(make_biased_shapley(0.25))
    check_uniform_prob_mass(shap)
    assert shap.n_terminals == 9
    assert all(pt.n_iset == 1 and pt.n_seq == 4 for pt in shap.players)
    print("[ok] biased shapley: 9 terminals, 1 infoset x 3 actions per player")

    # matching pennies: value 0, gap of uniform profile = 0
    mp = build_efg(MatrixGame([np.array([[1., -1.], [-1., 1.]]),
                               np.array([[-1., 1.], [1., -1.]])]))
    from src import nash_gap
    xs = [mp.realization(i, mp.uniform_sigma(i)) for i in range(2)]
    gap, _ = nash_gap(mp, xs)
    assert abs(gap) < 1e-12, gap
    print("[ok] matching pennies: uniform profile has NashGap 0")

    print("ALL PASSED: test_games")


if __name__ == "__main__":
    main()
