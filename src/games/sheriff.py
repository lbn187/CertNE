"""Sheriff of Nottingham (simplified bargaining variant).

General-sum two-player EFG benchmark of Farina, Ling, Fang & Sandholm,
"Correlation in Extensive-Form Games: Saddle-Point Formulation and
Benchmarks" (NeurIPS 2019), formal description in arXiv:1905.12564 App. F.1.
Note the conference version (p.8) has a typo in the accept payoff
(``p*n - b_r``); the correct form, cross-checked against their Table 1
numbers, is ``v*n - b_r`` and is what we implement.

Rules: the Smuggler (player 0) secretly loads n in {0..n_max} items; then
r bargaining rounds follow, each round the Smuggler offers a bribe
b_t in {0..b_max} and the Sheriff (player 1) answers yes/no. Only the last
round is binding: yes = accept bribe and pass, no = inspect. Payoffs
(Smuggler, Sheriff): accept -> (v*n - b_r, b_r); inspect with n >= 1 ->
(-p*n, p*n); inspect with n = 0 -> (+s, -s). All actions are public except
the cargo choice, so the Sheriff's infoset drops n. No chance nodes.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

__all__ = ["SheriffRules", "SheriffGame", "make_sheriff"]


@dataclass(frozen=True)
class SheriffRules:
    value: float = 5.0
    penalty: float = 1.0
    compensation: float = 1.0
    n_max: int = 5
    b_max: int = 2
    n_rounds: int = 2


class _SheriffState:
    __slots__ = ("game", "cargo", "bribes", "answers")

    def __init__(self, game: "SheriffGame", cargo: Optional[int],
                 bribes: Tuple[int, ...], answers: Tuple[str, ...]) -> None:
        self.game = game
        self.cargo = cargo
        self.bribes = bribes
        self.answers = answers

    def is_terminal(self) -> bool:
        return len(self.answers) == self.game.rules.n_rounds

    def is_chance(self) -> bool:
        return False

    def chance_outcomes(self):
        raise RuntimeError("sheriff has no chance nodes")

    def player(self) -> int:
        if self.cargo is None or len(self.bribes) == len(self.answers):
            return 0
        return 1

    def infoset_key(self) -> Any:
        if self.player() == 0:
            return ("S", self.cargo, self.bribes, self.answers)
        return ("H", self.bribes, self.answers)

    def legal_actions(self) -> Sequence[Any]:
        rules = self.game.rules
        if self.cargo is None:
            return tuple(range(rules.n_max + 1))
        if len(self.bribes) == len(self.answers):
            return tuple(range(rules.b_max + 1))
        return ("y", "n")

    def child(self, action: Any) -> "_SheriffState":
        if self.cargo is None:
            return _SheriffState(self.game, int(action), self.bribes,
                                 self.answers)
        if len(self.bribes) == len(self.answers):
            return _SheriffState(self.game, self.cargo,
                                 self.bribes + (int(action),), self.answers)
        return _SheriffState(self.game, self.cargo, self.bribes,
                             self.answers + (str(action),))

    def utility(self) -> List[float]:
        rules = self.game.rules
        n = self.cargo
        b_last = self.bribes[-1]
        if self.answers[-1] == "y":
            return [rules.value * n - b_last, float(b_last)]
        if n >= 1:
            return [-rules.penalty * n, rules.penalty * n]
        return [rules.compensation, -rules.compensation]


class SheriffGame:
    n_players = 2

    def __init__(self, rules: SheriffRules) -> None:
        self.rules = rules

    def initial_state(self) -> _SheriffState:
        return _SheriffState(self, None, (), ())


def make_sheriff(value: float, penalty: float, compensation: float,
                 n_max: int = 5, b_max: int = 2,
                 n_rounds: int = 2) -> SheriffGame:
    return SheriffGame(SheriffRules(value, penalty, compensation,
                                    n_max, b_max, n_rounds))
