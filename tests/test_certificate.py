"""Pytest coverage for the KL/Pinsker runtime certificate."""

from pathlib import Path
import sys

import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import PokerGame, kuhn_rules, make_biased_shapley  # noqa: E402


@pytest.mark.parametrize("game_name,alg,weight_exp", [
    ("shapley", "rm+", 0.0),
    ("shapley", "dprm+", 2.0),
    ("kuhn2", "rm+", 2.0),
    ("kuhn2", "dprm+", 0.0),
])
def test_runtime_certificate_upper_bounds_nash_gap(
    game_name: str, alg: str, weight_exp: float,
) -> None:
    """The measurable certificate is sound at every sampled checkpoint."""
    game = (make_biased_shapley(0.25) if game_name == "shapley"
            else PokerGame(kuhn_rules(2)))
    efg = build_efg(game)
    out = run_cfr(efg, CFRConfig(
        alg=alg, t_max=256, checkpoints=(32, 128, 256),
        weight_exp=weight_exp,
    ))
    history = out["history"]

    for gap, certificate in zip(history["nash_gap"], history["certificate"]):
        assert certificate >= gap - 1e-9
