"""Hand-computed action-sequence traces through the poker rules engine,
plus exact Leduc2 tree-size regression pins.

Traces are verified against by-hand pot arithmetic, independent of the EFG
flattening: we drive PokerState directly via chance_outcomes()/child().
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import build_efg
from src.games import PokerGame, kuhn_rules, leduc_rules, make_biased_leduc


def deal(state, card):
    """Advance a chance node by dealing a specific card."""
    assert state.is_chance(), "expected a chance node"
    for child, _ in state.chance_outcomes():
        if child.deals[-1] == card:
            return child
    raise AssertionError(f"card {card} not available")


def play(game, cards, actions):
    """Deal `cards` in order, then apply `actions` (dealing between rounds)."""
    state = game.initial_state()
    cards = list(cards)
    for a in actions:
        while state.is_chance():
            state = deal(state, cards.pop(0))
        assert a in state.legal_actions(), (a, state.legal_actions())
        state = state.child(a)
    while state.is_chance() and cards:
        state = deal(state, cards.pop(0))
    assert state.is_terminal(), "trace did not reach a terminal"
    assert not cards, f"unused cards {cards}"
    return state


def approx(xs, ys):
    assert len(xs) == len(ys) and all(abs(a - b) < 1e-12 for a, b in zip(xs, ys)), \
        (xs, ys)


def main() -> None:
    # --- 3-player Kuhn (4 ranks, 1 suit, ante 1, bet 1, cap 1) -------------
    k3 = PokerGame(kuhn_rules(3))

    # P0 bets, P1 folds, P2 calls; showdown P0(r3) beats P2(r1).
    # contrib=(2,1,2), pot=5 -> util = (5-2, -1, -2) = (+3, -1, -2).
    approx(play(k3, [3, 0, 1], "rfc").utility(), [3.0, -1.0, -2.0])

    # P0 checks, P1 bets, P2 folds, P0 folds; P1 wins uncontested.
    # contrib=(1,2,1), pot=4 -> util = (-1, 4-2, -1) = (-1, +2, -1).
    approx(play(k3, [3, 0, 1], "crff").utility(), [-1.0, 2.0, -1.0])

    # All check; showdown among all three, highest rank (P0=r3) wins pot 3.
    approx(play(k3, [3, 0, 1], "ccc").utility(), [2.0, -1.0, -1.0])
    print("[ok] kuhn3 traces: bet-fold-call / check-bet-fold-fold / check-around")

    # --- 2-player Leduc (3 ranks x 2 suits, ante 1, bets 2/4, cap 2) -------
    l2 = PokerGame(leduc_rules(2))

    # Round 1: P0 raise 2, P1 reraise (level 4), P0 call -> 4 each.
    # Round 2: P0 check, P1 raise 4, P0 call -> 4 each. Board pairs P1's rank.
    # cards: P0 = rank2 (card 4), P1 = rank0 (card 0), board = rank0' (card 1).
    # contrib = ante 1 + 4 + 4 = 9 each, pot 18; P1 pairs -> P1 +9, P0 -9.
    approx(play(l2, [4, 0, 1], "rrc" + "crc").utility(), [-9.0, 9.0])

    # Fold to a round-1 raise: P0 raises to 2 (contrib 3), P1 folds (contrib 1).
    # (Hand ends before the board card, so only the two private cards are dealt.)
    approx(play(l2, [4, 0], "rf").utility(), [1.0, -1.0])
    print("[ok] leduc2 traces: raise-war showdown / fold to raise")

    # --- biased Leduc tie (beta = 0.25) ------------------------------------
    # Both hold rank0 (cards 0, 1), board rank1 (card 2), check-down.
    # Standard tie would refund contrib 1 each (util 0); biased pays each
    # tied player beta * contrib = 0.25, i.e. util = 0.25 - 1 = -0.75.
    b2 = make_biased_leduc(0.25)
    approx(play(b2, [0, 1, 2], "cc" + "cc").utility(), [-0.75, -0.75])

    # Non-tie terminals must be unchanged relative to standard Leduc.
    approx(play(b2, [4, 0], "rf").utility(), [1.0, -1.0])
    approx(play(b2, [4, 0, 1], "rrc" + "crc").utility(), [-9.0, 9.0])
    print("[ok] biased leduc: tie pays beta*wager, non-ties unchanged")

    # --- Leduc2 exact tree size (regression pin) ---------------------------
    efg = build_efg(PokerGame(leduc_rules(2)))
    assert efg.n_terminals == 5520, efg.n_terminals
    assert all(pt.n_iset == 468 for pt in efg.players), \
        [pt.n_iset for pt in efg.players]
    print(f"[ok] leduc2 tree: terminals={efg.n_terminals}, "
          f"infosets={[pt.n_iset for pt in efg.players]}")

    print("ALL PASSED: test_trace")


if __name__ == "__main__":
    main()
