"""General-sum Goofspiel (game of pure strategy), full-revelation variant.

GS(N): both players hold bid cards {1..N}; in round t they simultaneously
bid one remaining card for prize ``prizes[t]``. The higher bid wins the
prize; on a tie the prize *evaporates* (nobody scores), which makes the
game general-sum -- the same money-destruction mechanism as the biased
Leduc tie rule, in a structurally unrelated simultaneous-move game.

Simultaneity is encoded sequentially: player 0 bids (hidden), player 1
bids, then both bids are revealed. After each round the full bid history
is public (full revelation), so an infoset is the completed-round history;
player 1's infoset additionally excludes player 0's in-flight bid. The
prize sequence is an instance parameter (no chance nodes).
"""

from typing import Any, List, Optional, Sequence, Tuple

__all__ = ["GoofspielGame", "make_goofspiel"]


class _GoofState:
    __slots__ = ("game", "history", "pending")

    def __init__(self, game: "GoofspielGame",
                 history: Tuple[Tuple[int, int], ...],
                 pending: Optional[int]) -> None:
        self.game = game
        self.history = history
        self.pending = pending

    def is_terminal(self) -> bool:
        return len(self.history) == self.game.n_cards

    def is_chance(self) -> bool:
        return False

    def chance_outcomes(self):
        raise RuntimeError("goofspiel has no chance nodes")

    def player(self) -> int:
        return 0 if self.pending is None else 1

    def infoset_key(self) -> Any:
        return ("A", self.history) if self.pending is None \
            else ("B", self.history)

    def legal_actions(self) -> Sequence[int]:
        idx = self.player()
        used = frozenset(pair[idx] for pair in self.history)
        return tuple(c for c in range(self.game.n_cards) if c not in used)

    def child(self, action: int) -> "_GoofState":
        if self.pending is None:
            return _GoofState(self.game, self.history, int(action))
        return _GoofState(self.game,
                          self.history + ((self.pending, int(action)),), None)

    def utility(self) -> List[float]:
        scores = [0.0, 0.0]
        for t, (a, b) in enumerate(self.history):
            if a > b:
                scores[0] += self.game.prizes[t]
            elif b > a:
                scores[1] += self.game.prizes[t]
            # tie: prize evaporates (general-sum)
        return scores


class GoofspielGame:
    n_players = 2

    def __init__(self, prizes: Sequence[float]) -> None:
        self.prizes = tuple(float(v) for v in prizes)
        self.n_cards = len(self.prizes)

    def initial_state(self) -> _GoofState:
        return _GoofState(self, (), None)


def make_goofspiel(prizes: Sequence[float]) -> GoofspielGame:
    return GoofspielGame(prizes)
