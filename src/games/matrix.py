"""Normal-form (matrix) games wrapped as one-infoset-per-player EFGs."""

from typing import Any, List, Sequence, Tuple

import numpy as np

__all__ = ["MatrixGame", "make_biased_shapley", "random_general_sum"]


class _MatrixState:
    __slots__ = ("game", "chosen")

    def __init__(self, game: "MatrixGame", chosen: Tuple[int, ...]) -> None:
        self.game = game
        self.chosen = chosen

    def is_terminal(self) -> bool:
        return len(self.chosen) == self.game.n_players

    def is_chance(self) -> bool:
        return False

    def player(self) -> int:
        return len(self.chosen)

    def infoset_key(self) -> Any:
        return "root"

    def legal_actions(self) -> Sequence[int]:
        return range(self.game.shape[len(self.chosen)])

    def child(self, action: int) -> "_MatrixState":
        return _MatrixState(self.game, self.chosen + (action,))

    def chance_outcomes(self):
        raise RuntimeError("matrix games have no chance nodes")

    def utility(self) -> List[float]:
        return [float(p[self.chosen]) for p in self.game.payoffs]


class MatrixGame:
    """n-player normal-form game; players 'move' sequentially but blindly."""

    def __init__(self, payoffs: Sequence[np.ndarray]) -> None:
        self.payoffs = [np.asarray(p, dtype=float) for p in payoffs]
        self.n_players = len(self.payoffs)
        self.shape = self.payoffs[0].shape
        for p in self.payoffs:
            if p.shape != self.shape or p.ndim != self.n_players:
                raise ValueError("payoff tensors must share shape (a_1..a_n)")

    def initial_state(self) -> _MatrixState:
        return _MatrixState(self, ())


def make_biased_shapley(eta: float) -> MatrixGame:
    """Biased Shapley game of the NPCFR paper: bonus eta to both at (a0, b2)."""
    a = np.eye(3)
    b = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    a[0, 2] += eta
    b[0, 2] += eta
    return MatrixGame([a, b])


def random_general_sum(rng: np.random.Generator,
                       shape: Tuple[int, ...]) -> MatrixGame:
    """Random general-sum game with i.i.d. U(0,1) payoffs (for stress tests)."""
    return MatrixGame([rng.uniform(0.0, 1.0, size=shape) for _ in shape])
