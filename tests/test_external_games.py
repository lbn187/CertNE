"""External-validity games: Sheriff + general-sum Goofspiel invariants.

Checks (a) tree counts against the closed-form formulas validated versus
Farina et al. 2019 Table 4, (b) hand-computed payoffs, (c) both games are
genuinely general-sum, (d) formal certificate checks on a short CFR run.
Run directly: python tests/test_external_games.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import build_efg
from src.cfr import CFRConfig, run_cfr
from src.games import make_goofspiel, make_sheriff
from tests.test_games import check_uniform_prob_mass


def sheriff_counts(n_max: int, b_max: int, r: int):
    """Closed-form counts cross-checked against Farina et al. Table 4."""
    sheriff_isets = sum((b_max + 1) ** t * 2 ** (t - 1) for t in range(1, r + 1))
    smug_isets = 1 + sum((n_max + 1) * ((b_max + 1) * 2) ** (t - 1)
                         for t in range(1, r + 1))
    terminals = (n_max + 1) * (2 * (b_max + 1)) ** r
    return smug_isets, sheriff_isets, terminals


def check_sheriff_structure() -> None:
    for n_max, b_max, r in [(3, 2, 1), (5, 2, 2), (3, 3, 3)]:
        efg = build_efg(make_sheriff(5.0, 1.0, 1.0, n_max, b_max, r))
        smug, sher, term = sheriff_counts(n_max, b_max, r)
        assert efg.n_terminals == term, (n_max, b_max, r, efg.n_terminals)
        assert efg.players[0].n_iset == smug, (efg.players[0].n_iset, smug)
        assert efg.players[1].n_iset == sher, (efg.players[1].n_iset, sher)
        # sequence counts: 1 (empty) + total actions across infosets
        smug_seq = 1 + (n_max + 1) + (smug - 1) * (b_max + 1)
        assert efg.players[0].n_seq == smug_seq, efg.players[0].n_seq
        assert efg.players[1].n_seq == 1 + 2 * sher, efg.players[1].n_seq
        check_uniform_prob_mass(efg)
        print(f"[ok] sheriff({n_max},{b_max},{r}): terminals={term}, "
              f"infosets=({smug},{sher})")
    # Table 4 anchor (b_max=3, r=5): Sheriff sequences = 37,448 excl. empty
    _, sher, _ = sheriff_counts(6, 3, 5)
    assert 2 * sher == 37448, 2 * sher
    print("[ok] sheriff Table-4 anchor: 2*isets(b_max=3,r=5) == 37448")


def check_sheriff_payoffs() -> None:
    game = make_sheriff(5.0, 1.0, 1.0, n_max=3, b_max=2, n_rounds=1)
    s = game.initial_state()
    # n=2, bribe=2, yes -> (v*n - b, b) = (8, 2)  [Farina Table 1 SW-max]
    t = s.child(2).child(2).child("y")
    assert t.is_terminal() and t.utility() == [8.0, 2.0], t.utility()
    # n=2, bribe=0, no -> (-p*n, p*n) = (-2, 2)
    t = s.child(2).child(0).child("n")
    assert t.utility() == [-2.0, 2.0], t.utility()
    # n=0, bribe=1, no -> (+s, -s) = (1, -1)
    t = s.child(0).child(1).child("n")
    assert t.utility() == [1.0, -1.0], t.utility()
    # n=0, bribe=1, yes -> (0*v - 1, 1) = (-1, 1)
    t = s.child(0).child(1).child("y")
    assert t.utility() == [-1.0, 1.0], t.utility()
    print("[ok] sheriff payoffs match hand-computed cases (incl. Table-1 pt)")


def check_goofspiel() -> None:
    # N=2, prizes (1,2): all four leaves hand-computed
    efg = build_efg(make_goofspiel([1.0, 2.0]))
    assert efg.n_terminals == 4, efg.n_terminals
    game = make_goofspiel([1.0, 2.0])
    s = game.initial_state()
    assert s.child(0).child(0).child(1).child(1).utility() == [0.0, 0.0]
    assert s.child(0).child(1).child(1).child(0).utility() == [2.0, 1.0]
    assert s.child(1).child(0).child(0).child(1).utility() == [1.0, 2.0]
    check_uniform_prob_mass(efg)
    # N=5: terminal count (5!)^2, infoset player-1 excludes in-flight bid
    efg5 = build_efg(make_goofspiel([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert efg5.n_terminals == 14400, efg5.n_terminals
    check_uniform_prob_mass(efg5)
    print(f"[ok] goofspiel: N=2 leaves hand-checked, N=5 terminals=14400, "
          f"infosets={[pt.n_iset for pt in efg5.players]}")


def check_general_sum() -> None:
    for name, game in [
        ("sheriff", make_sheriff(5.0, 1.0, 1.0, 3, 2, 1)),
        ("goofspiel", make_goofspiel([1.0, 2.0, 3.0])),
    ]:
        efg = build_efg(game)
        dev = np.abs(efg.term_util.sum(axis=0)).max()
        assert dev > 1e-6, f"{name} looks zero-sum"
        print(f"[ok] {name} is general-sum (max |sum_i u_i| = {dev:.3f})")


def check_certificate_soundness() -> None:
    for name, game in [
        ("sheriff", make_sheriff(3.2, 1.5, 0.8, 5, 2, 2)),
        ("goofspiel", make_goofspiel([2.1, 0.9, 3.4, 1.2, 4.8])),
    ]:
        efg = build_efg(game)
        for update in ("simultaneous", "alternating"):
            strict = update == "alternating"
            res = run_cfr(efg, CFRConfig(
                alg="pcfr+", t_max=256, eval_every=32, weight_exp=2.0,
                update=update, strict_regret_logging=strict,
            ))
            hist = res["history"]
            field = "certificate_tight_strict" if strict else "certificate_tight"
            assert np.all(np.isfinite(hist["certificate_tight"]))
            for cert, gap, t in zip(hist[field], hist["nash_gap"], hist["t"]):
                assert cert >= gap - 1e-9, (name, update, t, cert, gap)
        print(f"[ok] {name}: formal certificate >= nash_gap at every "
              f"checkpoint (simultaneous or strict alternating)")


def main() -> None:
    check_sheriff_structure()
    check_sheriff_payoffs()
    check_goofspiel()
    check_general_sum()
    check_certificate_soundness()
    print("all external-game checks passed")


if __name__ == "__main__":
    main()
