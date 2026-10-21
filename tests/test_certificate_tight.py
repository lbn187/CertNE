"""Pytest coverage for the exact two-player certificate."""

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import MatrixGame, PokerGame, kuhn_rules, make_biased_leduc  # noqa: E402

TOL = 1e-9


@pytest.mark.parametrize("game_name,alg,update,weight_exp,strict", [
    ("kuhn2", "rm+", "simultaneous", 0.0, False),
    ("kuhn2", "pcfr+", "alternating", 1.0, True),
    ("kuhn2", "dprm+", "alternating", 2.0, True),
    ("biased_leduc2", "pcfr+", "simultaneous", 1.0, False),
])
def test_formal_tight_certificate_is_sound_and_no_looser_than_kl_bound(
    game_name: str, alg: str, update: str, weight_exp: float, strict: bool,
) -> None:
    game = (PokerGame(kuhn_rules(2)) if game_name == "kuhn2"
            else make_biased_leduc(0.25))
    efg = build_efg(game)
    t_max = 96 if game_name == "kuhn2" else 24
    out = run_cfr(efg, CFRConfig(
        alg=alg, update=update, weight_exp=weight_exp, lam=0.3,
        strict_regret_logging=strict,
        t_max=t_max, checkpoints=(t_max // 2, t_max),
    ))
    history = out["history"]
    gap = np.asarray(history["nash_gap"])
    field = "certificate_tight_strict" if strict else "certificate_tight"
    tight = np.asarray(history[field])
    loose = np.asarray(history["certificate"])

    assert np.all(tight >= gap - TOL)
    assert np.all(tight <= loose + TOL)


def test_alternating_non_strict_tight_score_has_fixed_unsound_counterexample() -> None:
    """Legacy alternating score is finite but heuristic; strict is formal."""
    payoffs = [
        np.array([
            [1.1435656192549395, -1.7220870097811383, -1.3420831723206557],
            [0.036566470040846344, 0.034009772344036904, -0.36806423170854297],
            [-1.2568133605284344, 0.3761192624261942, 1.0521536038637889],
        ]),
        np.array([
            [1.3415047626702208, -1.3220520068559716, 1.037375333160392],
            [0.46907004611484515, -1.6693321079583154, -0.293011050565688],
            [0.016062636001748842, -1.0599909894885902, -0.7186291118049777],
        ]),
    ]
    efg = build_efg(MatrixGame(payoffs))
    base = dict(
        alg="pcfr+", update="alternating", weight_exp=2.0,
        t_max=32, checkpoints=tuple(range(1, 33)),
    )

    heuristic = run_cfr(efg, CFRConfig(**base))["history"]
    formal = run_cfr(
        efg, CFRConfig(**base, strict_regret_logging=True))["history"]
    heuristic_score = np.asarray(heuristic["certificate_tight"])
    strict_score = np.asarray(formal["certificate_tight_strict"])
    gap = np.asarray(heuristic["nash_gap"])

    assert np.all(np.isfinite(heuristic_score))
    assert np.any(heuristic_score < gap - 1e-3)
    assert np.all(strict_score >= gap - TOL)


def test_tight_certificate_is_not_emitted_for_three_players() -> None:
    efg = build_efg(PokerGame(kuhn_rules(3)))
    out = run_cfr(efg, CFRConfig(
        alg="rm+", update="alternating", weight_exp=1.0,
        t_max=16, checkpoints=(8, 16),
    ))

    assert "certificate_tight" not in out["history"]
