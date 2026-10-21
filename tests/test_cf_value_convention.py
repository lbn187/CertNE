"""Pin the counterfactual-value convention behind the weighted regret decomposition.

The laminar identity used by Lemma (weighted regret decomposition) reads

    u_i(x', sigma_{-i}) - u_i(sigma)
        = sum_{I in I_i} pi^{x'}_i(I) sum_{a in A(I)} x'(a|I) ir_i(I, a),

and it holds only when the counterfactual action value

    v_i(I, a) = sum_{z in Z(I,a)} p_c(z) x_{-i}(z) pi_i(z | Ia) u_i(z)

carries player i's own continuation pi_i(z | Ia) below Ia while excluding its
own reach before I: the convention of Zinkevich et al. (2007, Theorem 3) and of
EFG.backward_sum. Dropping the continuation term breaks the identity. On a root
infoset I0 with actions A, B, where A leads to I1 with two actions that each
pay 1 and B pays 1.5, and sigma mixes at I1, the deviation x' = A gives
LHS = +0.5 whereas the continuation-free RHS gives -0.5.

The tests below run that check on real games: the identity holds to machine
precision under the engine's convention and fails by a wide margin without it.
"""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import build_efg  # noqa: E402
from src.games import (PokerGame, kuhn_rules, make_biased_leduc,  # noqa: E402
                       make_biased_shapley)


def _game(name: str):
    if name == "kuhn2":
        return PokerGame(kuhn_rules(2))
    if name == "leduc":
        return make_biased_leduc(0.25)
    if name == "shapley":
        return make_biased_shapley(0.25)
    raise ValueError(name)


def _random_sigma(efg, i: int, rng) -> np.ndarray:
    """A strictly positive behavior strategy for player i."""
    return efg.normalize(i, rng.random(efg.players[i].n_seq) + 1e-2)


def _worst_residual(efg, rng, own_continuation: bool, n_deviations: int = 8) -> float:
    """max over players and deviations of |LHS - RHS| for the laminar identity.

    ``own_continuation`` selects the convention under test: True is
    EFG.backward_sum (own continuation included below the action), False is the
    raw terminal accumulation (no own continuation), i.e. the definition an
    earlier draft of eq:cfv wrote.
    """
    sigmas = [_random_sigma(efg, j, rng) for j in range(efg.n_players)]
    xs = [efg.realization(j, sigmas[j]) for j in range(efg.n_players)]
    base = float(np.sum(efg.joint_reach(xs) * efg.term_util[0]))  # u_0(sigma)

    worst = 0.0
    for i in range(efg.n_players):
        pt = efg.players[i]
        term_w = efg.opp_reach(i, xs) * efg.term_util[i]
        if own_continuation:
            v = efg.backward_sum(i, sigmas[i], term_w)
        else:
            v = np.zeros(pt.n_seq)
            np.add.at(v, efg.term_seq[i], term_w)
        ir = efg.local_regret(i, sigmas[i], v)

        # u_i(sigma) for this player, then one value per deviating plan x'.
        u_sigma = float(np.sum(efg.joint_reach(xs) * efg.term_util[i]))
        opp = efg.opp_reach(i, xs)
        for _ in range(n_deviations):
            sigma_p = _random_sigma(efg, i, rng)
            xp = efg.realization(i, sigma_p)
            lhs = float(np.sum(opp * xp[efg.term_seq[i]] * efg.term_util[i])) - u_sigma
            rhs = float(np.sum(xp[1:] * ir[1:]))
            worst = max(worst, abs(lhs - rhs))
    return worst


@pytest.mark.parametrize("game_name", ["kuhn2", "shapley"])
def test_laminar_identity_holds_with_own_continuation(game_name: str) -> None:
    """EFG.backward_sum is the convention under which the identity is exact."""
    efg = build_efg(_game(game_name))
    assert _worst_residual(efg, np.random.default_rng(0), True) < 1e-10


@pytest.mark.parametrize("game_name", ["kuhn2"])
def test_laminar_identity_fails_without_own_continuation(game_name: str) -> None:
    """Dropping own continuation below Ia breaks the identity (regression guard)."""
    efg = build_efg(_game(game_name))
    assert _worst_residual(efg, np.random.default_rng(0), False) > 1e-3


def _random_plan(efg, i: int, rng) -> np.ndarray:
    return efg.realization(i, _random_sigma(efg, i, rng))


def _u(efg, i: int, x_i: np.ndarray, prof: list) -> float:
    """u_i(x_i, prof_{-i}) over the terminal distribution."""
    r = efg.term_chance * x_i[efg.term_seq[i]]
    for j in range(efg.n_players):
        if j != i:
            r = r * prof[j][efg.term_seq[j]]
    return float(np.sum(r * efg.term_util[i]))


def _comparator_residual(efg, i: int, rng, p_c_power: int = 1,
                         n_deviations: int = 16) -> float:
    """max |LHS - RHS| of the comparator identity, with p_c applied p_c_power times.

    LHS = u_i(x', xbar_{-i}) - sum_t rho_t u_i(x', y^t_{-i}),  RHS = <x', w_i>,
    with the implemented weight w_i = p_c u_i (prod_{j!=i} xbar_j - coal).
    p_c_power=2 models a double-counted chance reach.
    """
    xb = [_random_plan(efg, j, rng) for j in range(efg.n_players)]
    ys = [[_random_plan(efg, j, rng) for j in range(efg.n_players)]
          for _ in range(3)]
    rho = np.array([0.2, 0.3, 0.5])

    def outer(prof):
        r = np.ones(efg.n_terminals)
        for j in range(efg.n_players):
            if j != i:
                r = r * prof[j][efg.term_seq[j]]
        return r

    prod_xb = outer(xb)
    coal = sum(rho[k] * outer(ys[k]) for k in range(len(ys)))
    w = efg.term_chance ** p_c_power * efg.term_util[i] * (prod_xb - coal)

    worst = 0.0
    for _ in range(n_deviations):
        xp = _random_plan(efg, i, rng)
        lhs = _u(efg, i, xp, xb) - sum(rho[k] * _u(efg, i, xp, ys[k])
                                       for k in range(len(ys)))
        rhs = float(np.sum(w * xp[efg.term_seq[i]]))
        worst = max(worst, abs(lhs - rhs))
    return worst


@pytest.mark.parametrize("game_name", ["leduc", "kuhn2"])
def test_comparator_identity_holds_for_all_deviations(game_name: str) -> None:
    """Both sides are the same linear functional of x', so agreement on random
    deviations covers every x' in X_i; no enumeration of pure strategies is needed."""
    efg = build_efg(_game(game_name))
    assert _comparator_residual(efg, 0, np.random.default_rng(1)) < 1e-10


@pytest.mark.parametrize("game_name", ["leduc", "kuhn2"])
def test_comparator_identity_has_teeth_against_double_counted_chance(
        game_name: str) -> None:
    """Applying p_c twice must break the identity, so the check above is not vacuous."""
    efg = build_efg(_game(game_name))
    assert _comparator_residual(efg, 0, np.random.default_rng(1),
                                p_c_power=2) > 1e-6
