"""Parameterized n-player limit poker covering Kuhn and Leduc variants.

One implementation covers:
  - Kuhn poker      : 1 round, no board, 1 suit, raise cap 1, bet 1
  - Leduc hold'em   : 2 rounds, 1 board card, 2 suits, raise cap 2, bets (2, 4)
  - 3-player Kuhn / Leduc : same rules with n_players=3 (deck grows by rank)

Cards are integers ``rank * n_suits + suit``. Chance deals cards one at a
time (private cards in seat order, then one board card at the start of each
round after the first). Betting is standard limit poker: 'f' fold, 'c'
check/call, 'r' bet/raise, with at most ``max_raises`` raises per round.

A ``bias_fn`` hook on PokerGame adds a per-player payoff adjustment at
terminals, turning the (zero-sum) poker game into a general-sum benchmark
(e.g. the biased Leduc of the NPCFR paper — exact form in
notes/PAPER_BENCHMARKS.md).
"""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

__all__ = ["PokerRules", "PokerGame", "PokerState", "kuhn_rules", "leduc_rules"]


@dataclass(frozen=True)
class PokerRules:
    n_players: int = 2
    n_ranks: int = 3
    n_suits: int = 2
    n_rounds: int = 2
    max_raises: int = 2
    bet_sizes: Tuple[int, ...] = (2, 4)
    ante: int = 1


def kuhn_rules(n_players: int = 2) -> PokerRules:
    """Kuhn poker; the classic n-player variant uses n_players+1 ranks."""
    return PokerRules(n_players=n_players, n_ranks=n_players + 1, n_suits=1,
                      n_rounds=1, max_raises=1, bet_sizes=(1,), ante=1)


def leduc_rules(n_players: int = 2) -> PokerRules:
    """Leduc hold'em; deck has n_players+1 ranks x 2 suits."""
    return PokerRules(n_players=n_players, n_ranks=n_players + 1, n_suits=2,
                      n_rounds=2, max_raises=2, bet_sizes=(2, 4), ante=1)


class PokerState:
    __slots__ = ("game", "deals", "round_idx", "history", "folded", "contrib",
                 "paid", "level", "pending", "raises", "done")

    def __init__(self, game: "PokerGame", deals: Tuple[int, ...],
                 round_idx: int, history: Tuple[str, ...],
                 folded: Tuple[bool, ...], contrib: Tuple[int, ...],
                 paid: Tuple[int, ...], level: int,
                 pending: Tuple[int, ...], raises: int, done: bool) -> None:
        self.game = game
        self.deals = deals
        self.round_idx = round_idx
        self.history = history
        self.folded = folded
        self.contrib = contrib
        self.paid = paid
        self.level = level
        self.pending = pending
        self.raises = raises
        self.done = done

    # -- structure ---------------------------------------------------------

    def is_terminal(self) -> bool:
        return self.done

    def _cards_needed(self) -> int:
        return self.game.rules.n_players + self.round_idx

    def is_chance(self) -> bool:
        return not self.done and len(self.deals) < self._cards_needed()

    def player(self) -> int:
        return self.pending[0]

    def infoset_key(self) -> Any:
        n = self.game.rules.n_players
        p = self.pending[0]
        board = self.deals[n:n + self.round_idx]
        return (self.deals[p], board, self.history)

    # -- chance ------------------------------------------------------------

    def chance_outcomes(self) -> List[Tuple["PokerState", float]]:
        remaining = [c for c in self.game.deck if c not in self.deals]
        prob = 1.0 / len(remaining)
        out = []
        for card in remaining:
            deals = self.deals + (card,)
            pending = self.pending
            if len(deals) == self._cards_needed():
                pending = tuple(
                    q for q in range(self.game.rules.n_players)
                    if not self.folded[q]
                )
            out.append((PokerState(
                self.game, deals, self.round_idx, self.history, self.folded,
                self.contrib, self.paid, self.level, pending, self.raises,
                self.done), prob))
        return out

    # -- betting -----------------------------------------------------------

    def legal_actions(self) -> Sequence[str]:
        rules = self.game.rules
        p = self.pending[0]
        owe = self.level - self.paid[p]
        can_raise = self.raises < rules.max_raises
        if owe > 0:
            return ("f", "c", "r") if can_raise else ("f", "c")
        return ("c", "r") if can_raise else ("c",)

    def child(self, action: str) -> "PokerState":
        rules = self.game.rules
        n = rules.n_players
        p = self.pending[0]
        history = self.history + (action,)
        folded = self.folded
        contrib = list(self.contrib)
        paid = list(self.paid)
        level = self.level
        raises = self.raises
        if action == "f":
            folded = folded[:p] + (True,) + folded[p + 1:]
            pending = self.pending[1:]
        elif action == "c":
            owe = level - paid[p]
            contrib[p] += owe
            paid[p] = level
            pending = self.pending[1:]
        elif action == "r":
            level += rules.bet_sizes[self.round_idx]
            contrib[p] += level - paid[p]
            paid[p] = level
            raises += 1
            order = [(p + k) % n for k in range(1, n)]
            pending = tuple(q for q in order if not folded[q])
        else:
            raise ValueError(action)
        round_idx = self.round_idx
        done = False
        if n - sum(folded) == 1:
            done = True
        elif not pending:
            if round_idx == rules.n_rounds - 1:
                done = True
            else:
                round_idx += 1
                paid = [0] * n
                level = 0
                raises = 0
                history = history + ("/",)
        return PokerState(self.game, self.deals, round_idx, history, folded,
                          tuple(contrib), tuple(paid), level, pending, raises,
                          done)

    # -- payoff ------------------------------------------------------------

    def _strength(self, i: int) -> Tuple[int, int]:
        rules = self.game.rules
        rank = self.deals[i] // rules.n_suits
        board = self.deals[rules.n_players:]
        pair = any(b // rules.n_suits == rank for b in board)
        return (1 if pair else 0, rank)

    def utility(self) -> List[float]:
        n = self.game.rules.n_players
        pot = float(sum(self.contrib))
        active = [i for i in range(n) if not self.folded[i]]
        if len(active) == 1:
            winners = active
        else:
            best = max(self._strength(i) for i in active)
            winners = [i for i in active if self._strength(i) == best]
        util = [-float(c) for c in self.contrib]
        share = pot / len(winners)
        for i in winners:
            util[i] += share
        if self.game.bias_fn is not None:
            bonus = self.game.bias_fn(self)
            util = [u + b for u, b in zip(util, bonus)]
        return util


class PokerGame:
    """Game definition consumed by src.efg.build_efg."""

    def __init__(self, rules: PokerRules,
                 bias_fn: Optional[Callable[[PokerState], Sequence[float]]] = None
                 ) -> None:
        deck_size = rules.n_ranks * rules.n_suits
        cards_used = rules.n_players + max(rules.n_rounds - 1, 0)
        if deck_size < cards_used:
            raise ValueError("deck too small for the requested game")
        if len(rules.bet_sizes) < rules.n_rounds:
            raise ValueError("need one bet size per round")
        self.rules = rules
        self.n_players = rules.n_players
        self.deck = tuple(range(deck_size))
        self.bias_fn = bias_fn

    def initial_state(self) -> PokerState:
        n = self.rules.n_players
        return PokerState(
            self, deals=(), round_idx=0, history=(),
            folded=(False,) * n, contrib=(self.rules.ante,) * n,
            paid=(0,) * n, level=0, pending=(), raises=0, done=False,
        )
