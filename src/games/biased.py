"""Biased (general-sum) Leduc of the NPCFR paper (arXiv:2504.18868).

Exact construction: in standard Leduc a
showdown tie returns each player's wager; in the biased variant each tied
player receives only ``beta`` times their wager (beta ~ U(0, 1/2)), so money
evaporates and the game becomes general-sum. Non-tie terminals are unchanged.

CAVEAT: with n_players >= 3 a folded player's dead money is
discarded on ties under this formula; the paper only defines the 2-player
variant, so n_players > 2 is not covered by the definition.
"""

from typing import List

from .poker import PokerGame, PokerState, leduc_rules

__all__ = ["make_biased_leduc"]


def make_biased_leduc(beta: float, n_players: int = 2) -> PokerGame:
    rules = leduc_rules(n_players)

    def bias_fn(state: PokerState) -> List[float]:
        n = rules.n_players
        active = [i for i in range(n) if not state.folded[i]]
        bonus = [0.0] * n
        if len(active) <= 1:
            return bonus  # fold win: unchanged
        best = max(state._strength(i) for i in active)
        winners = [i for i in active if state._strength(i) == best]
        if len(winners) <= 1:
            return bonus  # clear winner: unchanged
        # base utility gave each tied winner pot/|W| - contrib_i;
        # the biased game pays beta * contrib_i - contrib_i instead.
        pot = float(sum(state.contrib))
        share = pot / len(winners)
        for i in winners:
            bonus[i] = beta * state.contrib[i] - share
        return bonus

    return PokerGame(rules, bias_fn=bias_fn)
