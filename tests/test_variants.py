"""Arm-pool variants: bit-identity of the legacy algorithms, projection
correctness, literal-formula references on a matrix game, and soundness of
the profile-sequence certificate for every new algorithm.

Run: python3 tests/test_variants.py
"""

import importlib.util
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import CFRConfig, build_efg, run_cfr  # noqa: E402
from src.games import MatrixGame, PokerGame, kuhn_rules, make_biased_leduc  # noqa: E402
from src.variants import project_l1_floor, solve_l2_shift  # noqa: E402

REF_DIR = os.path.join(ROOT, "tests", "ref_src_20261001")
HIST_KEYS = ("t", "nash_gap", "certificate_seq", "certificate_seq_l1",
             "certificate_tight", "certificate_tight_strict", "certificate_coal")


def _load_ref():
    spec = importlib.util.spec_from_file_location(
        "ref_src", os.path.join(REF_DIR, "__init__.py"),
        submodule_search_locations=[REF_DIR])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ref_src"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_legacy_bit_identical():
    ref = _load_ref()
    games = [("leduc2", make_biased_leduc(0.3)), ("kuhn3", PokerGame(kuhn_rules(3)))]
    n_checked = 0
    for gname, game in games:
        for alg in ("rm+", "pcfr+", "dprm+"):
            for upd in ("simultaneous", "alternating"):
                kw = dict(alg=alg, update=upd, t_max=512, eval_every=128,
                          weight_exp=2.0, lam=0.3, strict_regret_logging=True)
                new = run_cfr(build_efg(game), CFRConfig(**kw))
                old = ref.run_cfr(ref.build_efg(game), ref.CFRConfig(**kw))
                for k in HIST_KEYS:
                    if k in old["history"]:
                        a, b = np.asarray(old["history"][k]), np.asarray(new["history"][k])
                        assert np.array_equal(a, b), (gname, alg, upd, k)
                        n_checked += 1
                for i in range(len(new["xs_bar"])):
                    assert np.array_equal(old["xs_bar"][i], new["xs_bar"][i]), (gname, alg, upd)
    assert n_checked >= 60, n_checked
    print(f"legacy bit-identical: {n_checked} history arrays compared")


def _simplex_proj_ref(y, floor):
    p = np.maximum(y, 0.0)
    if p.sum() >= floor:
        return p
    u = np.sort(y)[::-1]
    css = np.cumsum(u) - floor
    rho = np.nonzero(u - css / np.arange(1, len(y) + 1) > 0)[0][-1]
    theta = css[rho] / (rho + 1)
    return np.maximum(y - theta, 0.0)


def test_projections():
    efg = build_efg(make_biased_leduc(0.3))
    rng = np.random.default_rng(0)
    for i in range(2):
        pt = efg.players[i]
        assert pt.iset_start[0] == 1 and np.all(pt.iset_end[:-1] == pt.iset_start[1:])
        assert pt.iset_end[-1] == pt.n_seq
        for floor in (0.1, 1.0, 10.0):
            y = rng.normal(0.0, 3.0, size=pt.n_seq)
            got = project_l1_floor(efg, i, y, floor)
            for s, e in zip(pt.iset_start, pt.iset_end):
                want = _simplex_proj_ref(y[s:e], floor)
                assert np.allclose(got[s:e], want, atol=1e-12, rtol=0), (i, floor)
            assert got[0] == 0.0
        v = rng.normal(0.0, 3.0, size=pt.n_seq)
        target = rng.uniform(0.5, 5.0, size=pt.n_iset)
        need = np.ones(pt.n_iset, bool)
        g = solve_l2_shift(efg, i, v, target, need)
        for k, (s, e) in enumerate(zip(pt.iset_start, pt.iset_end)):
            got_sq = np.sum(np.maximum(v[s:e] - g[k], 0.0) ** 2)
            assert abs(got_sq - target[k]) < 1e-9 * target[k], (k, got_sq, target[k])
    print("projections: ok")


# ---------------------------------------------------------------- matrix refs
A = np.array([[3.0, 0.0, 1.0], [1.0, 4.0, 0.0], [0.0, 2.0, 5.0]])
B = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [4.0, 0.0, 1.0]])
T_REF = 300




def _play_ref(step0, step1, alternating):
    """Generic alternating/simultaneous driver for one-infoset references.
    step_i(u_vec, t) -> next strategy of player i given its utility vector."""
    x = np.full(3, 1 / 3)
    y = np.full(3, 1 / 3)
    xs, ys = [], []
    for t in range(1, T_REF + 1):
        xs.append(x.copy()); ys.append(y.copy())
        u0 = A @ y
        x_new = step0(u0, x, t)
        u1 = B.T @ (x_new if alternating else x)
        y_new = step1(u1, y, t)
        x, y = x_new, y_new
    return np.array(xs), np.array(ys)






def _dcfr_ref(alpha, beta, plus, predictive):
    def make():
        st = {"S": np.zeros(3)}

        def step(u, x, t):
            ir = u - u @ x
            ta = float(t) ** alpha
            da = ta / (ta + 1.0)
            if plus:
                r = np.maximum(st["S"] + ir, 0.0)
                st["S"] = r * da
                z = np.maximum(st["S"] + ir, 0.0) if predictive else r
            else:
                r = st["S"] + ir
                tb = float(t) ** beta
                st["S"] = r * np.where(r > 0, da, tb / (tb + 1.0))
                z = np.maximum(r, 0.0)
            return z / z.sum() if z.sum() > 0 else np.full(3, 1 / 3)
        return step
    return make


def _sprm_ref(eta, floor):
    def make():
        st = {"w": np.full(3, floor / 3)}

        def step(u, x, t):
            ir = u - u @ x
            st["w"] = _simplex_proj_ref(st["w"] + eta * ir, floor)
            z = _simplex_proj_ref(st["w"] + eta * ir, floor)
            return z / z.sum()
        return step
    return make


def _stprm_ref(eta, r0):
    def make():
        st = {"w": np.full(3, r0)}

        def step(u, x, t):
            ir = u - u @ x
            w = np.maximum(st["w"] + eta * ir, 0.0)
            pred = ir
            if np.all(w <= r0):
                w = np.full(3, r0); pred = np.zeros(3)
            st["w"] = w
            z = np.maximum(w + eta * pred, 0.0)
            return z / z.sum() if z.sum() > 0 else np.full(3, 1 / 3)
        return step
    return make


def _irprm_ref():
    def make():
        st = {"rt": np.zeros(3), "r": np.zeros(3), "m": np.zeros(3),
              "xt": np.full(3, 1 / 3)}

        def nxt(m):
            if np.all(st["rt"] <= 0):
                st["m"] = np.zeros(3); st["r"] = st["rt"].copy()
                return st["xt"].copy()
            c = np.linalg.norm(st["rt"])
            v = st["rt"] + m
            lo, hi = v.min() - c, v.max()
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if np.linalg.norm(np.maximum(v - mid, 0.0)) > c:
                    lo = mid
                else:
                    hi = mid
            st["m"] = m.copy(); st["r"] = v - 0.5 * (lo + hi)
            z = np.maximum(st["r"], 0.0)
            return z / z.sum()

        def step(u, x, t):
            d = u - st["m"]
            g = d - d @ x
            st["rt"] = np.maximum(st["r"] + g, 0.0)
            st["xt"] = (st["rt"] / st["rt"].sum() if st["rt"].sum() > 0
                        else np.full(3, 1 / 3))
            return nxt(u)
        return step
    return make




def test_matrix_references():
    cases = [
        ("dcfr", dict(alg="dcfr", disc_alpha=1.5, disc_beta=0.0), _dcfr_ref(1.5, 0.0, False, False)),
        ("dcfr+", dict(alg="dcfr+", disc_alpha=1.5), _dcfr_ref(1.5, 0.0, True, False)),
        ("pdcfr+", dict(alg="pdcfr+", disc_alpha=2.3), _dcfr_ref(2.3, 0.0, True, True)),
        ("sprm+", dict(alg="sprm+", step_eta=0.5, smooth_floor=1.0), _sprm_ref(0.5, 1.0)),
        ("stprm+", dict(alg="stprm+", step_eta=0.5, stable_r0=1.0), _stprm_ref(0.5, 1.0)),
        ("irprm+", dict(alg="irprm+"), _irprm_ref()),
    ]
    for name, kw, make in cases:
        for upd in ("alternating", "simultaneous"):
            efg = build_efg(MatrixGame([A, B]))
            cfg = CFRConfig(t_max=T_REF, eval_every=T_REF, weight_exp=0.0,
                            update=upd, **kw)
            out = run_cfr(efg, cfg)
            xs_ref, ys_ref = _play_ref(make(), make(), alternating=(upd == "alternating"))
            # uniform weights: xs_bar is the plain average of played strategies
            assert np.allclose(out["xs_bar"][0][1:], xs_ref.mean(axis=0), atol=1e-10), (name, upd)
            assert np.allclose(out["xs_bar"][1][1:], ys_ref.mean(axis=0), atol=1e-10), (name, upd)
    print("matrix references: ok (6 algorithms x 2 update orders)")


def test_hs_weights_reduce_to_power():
    efg = build_efg(PokerGame(kuhn_rules(2)))
    a = run_cfr(efg, CFRConfig(alg="pcfr+", update="alternating", t_max=300,
                               eval_every=100, weight_exp=2.0))
    b = run_cfr(efg, CFRConfig(alg="pcfr+", update="alternating", t_max=300,
                               eval_every=100, weight_schedule="hs",
                               hs_gamma0=2.0, hs_gamma_slope=0.0))
    for i in range(2):
        assert np.allclose(a["xs_bar"][i], b["xs_bar"][i], rtol=1e-12, atol=0), i
    assert np.allclose(a["history"]["nash_gap"], b["history"]["nash_gap"], rtol=1e-9)
    print("hs weights with constant gamma=2 reproduce weight_exp=2: ok")


def test_soundness_and_convergence():
    variants = [
        dict(alg="dcfr", update="alternating", weight_exp=2.0),
        dict(alg="dcfr+", update="alternating", weight_exp=4.0),
        dict(alg="pdcfr+", update="alternating", disc_alpha=2.3, weight_exp=5.0),
        dict(alg="pdcfr+", update="simultaneous", disc_alpha=2.3, weight_exp=5.0),
        dict(alg="dcfr", update="alternating", disc_schedule="hs", weight_schedule="hs"),
        dict(alg="pcfr+", update="alternating", weight_schedule="hs", hs_gamma0=15.0),
        dict(alg="sprm+", update="alternating", weight_exp=2.0),
        dict(alg="stprm+", update="alternating", weight_exp=2.0),
        dict(alg="exrm+", update="simultaneous", weight_exp=2.0),
        dict(alg="irprm+", update="alternating", weight_exp=2.0),
        dict(alg="irprm+", update="simultaneous", weight_exp=1.0),
    ]
    for game_name, game, t_max, tol in (("kuhn2", PokerGame(kuhn_rules(2)), 3000, 2e-2),
                                         ("kuhn3", PokerGame(kuhn_rules(3)), 1500, 5e-2)):
        for kw in variants:
            efg = build_efg(game)
            out = run_cfr(efg, CFRConfig(t_max=t_max, eval_every=t_max // 10,
                                         strict_regret_logging=True, **kw))
            h = out["history"]
            gap = np.asarray(h["nash_gap"]); cert = np.asarray(h["certificate_seq"])
            assert np.all(cert >= gap - 1e-12), (game_name, kw, (cert - gap).min())
            assert np.all(np.isfinite(cert)), (game_name, kw)
            assert gap.min() < tol, (game_name, kw, gap.min())
    print("soundness (certificate_seq >= gap at every checkpoint) and convergence: ok")


if __name__ == "__main__":
    test_projections()
    test_matrix_references()
    test_hs_weights_reduce_to_power()
    test_legacy_bit_identical()
    test_soundness_and_convergence()
    print("ALL PASSED: test_variants")
