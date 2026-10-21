"""Finite-difference check of the analytic KL decorrelation gradient.

Builds Kuhn poker, accumulates a few random past iterates into (J, avg),
then compares decorr_bias(own_reach='include') * w/(W+w) against a central
finite difference of decorr_loss at a random interior strategy profile.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import build_efg
from src.decorr import decorr_bias, decorr_loss
from src.games import PokerGame, kuhn_rules, make_biased_shapley

SEED = 42
FD_EPS = 1e-6


def random_profile(efg, rng):
    return [efg.normalize(i, rng.uniform(0.2, 1.0, efg.players[i].n_seq))
            for i in range(efg.n_players)]


def check_game(efg, name: str, rng) -> None:
    # accumulate 3 random past iterates with quadratic-ish weights
    j_sum = np.zeros(efg.n_terminals)
    avg = [np.zeros(pt.n_seq) for pt in efg.players]
    w_sum = 0.0
    for k in range(3):
        sigma = random_profile(efg, rng)
        xs = [efg.realization(i, sigma[i]) for i in range(efg.n_players)]
        w = float(k + 1) ** 2
        j_sum += w * efg.joint_reach(xs)
        for i in range(efg.n_players):
            avg[i] += w * xs[i]
        w_sum += w

    sigma = random_profile(efg, rng)
    w = 16.0
    xs = [efg.realization(i, sigma[i]) for i in range(efg.n_players)]
    grads = decorr_bias(efg, xs, sigma, j_sum, avg, w_sum, w,
                        own_reach="include")
    scale = w / (w_sum + w)  # decorr_bias is scaled by (W+w)/w

    max_rel = 0.0
    for i in range(efg.n_players):
        analytic = grads[i] * scale
        for s in range(1, efg.players[i].n_seq):
            sig_hi = sigma[i].copy()
            sig_hi[s] += FD_EPS
            sig_lo = sigma[i].copy()
            sig_lo[s] -= FD_EPS
            xs_hi = list(xs)
            xs_lo = list(xs)
            xs_hi[i] = efg.realization(i, sig_hi)
            xs_lo[i] = efg.realization(i, sig_lo)
            l_hi = decorr_loss(efg, xs_hi, j_sum, avg, w_sum, w)
            l_lo = decorr_loss(efg, xs_lo, j_sum, avg, w_sum, w)
            numeric = (l_hi - l_lo) / (2.0 * FD_EPS)
            denom = max(abs(numeric), abs(analytic[s]), 1e-6)
            rel = abs(numeric - analytic[s]) / denom
            max_rel = max(max_rel, rel)
            assert rel < 1e-4, (
                f"{name} player {i} seq {s}: "
                f"analytic={analytic[s]:.8e} numeric={numeric:.8e} rel={rel:.2e}"
            )
    print(f"[ok] {name}: FD gradient check passed, max rel err = {max_rel:.2e}")


def main() -> None:
    rng = np.random.default_rng(SEED)
    check_game(build_efg(make_biased_shapley(0.25)), "biased_shapley", rng)
    check_game(build_efg(PokerGame(kuhn_rules(2))), "kuhn2", rng)
    check_game(build_efg(PokerGame(kuhn_rules(3))), "kuhn3", rng)
    print("ALL PASSED: test_decorr_grad")


if __name__ == "__main__":
    main()
