"""Local regret-minimizer variants for the arm-pool extension.

Each class holds ONE player's per-sequence state and returns the behavioural
strategy for the next iteration. The certificate accumulators
(``regret_cum``, ``d_seen``, ``coal_seen``, ``avg``) live in ``cfr.run_cfr``
and never read this state: the certificate only needs the instantaneous
regret at the profile actually played, which the main loop computes before
calling ``step``.

Update rules (t is the 1-indexed iteration whose regret ``ir`` was just
observed; ``v`` is the counterfactual action-value vector at that iteration).

DCFR (Brown & Sandholm 2019; alpha, beta, gamma = 1.5, 0, 2):
    R_t = R_{t-1} (.) d_{t-1} + r_t,  d_t[a] = t^a/(t^a+1) if R_t[a] > 0
                                                 else t^b/(t^b+1)
    sigma_{t+1} ~ [R_t]^+
DCFR+ / PDCFR+ (Xu et al. 2024, Table 1):
    R_t = [R_{t-1} (t-1)^a/((t-1)^a+1) + r_t]^+
    DCFR+ : sigma_{t+1} ~ R_t
    PDCFR+: sigma_{t+1} ~ [R_t t^a/(t^a+1) + r_t]^+
Both are stored as S_t = R_t * d_t (the already-discounted state), so
R_{t+1} = [S_t + r_{t+1}]^+ and the PDCFR+ prediction is S_t + r_t.
Hyperparameter schedules (Zhang, McAleer & Sandholm 2026, eq. 4), with n the
total number of iterations: a_t = 1 + 3t/n, b_t = -1 - 2t/n.

Smooth PRM+ (Farina et al. 2023, Alg. 2), decision set X> = {R >= 0,
||R||_1 >= floor} per infoset, Pi = Euclidean projection onto X>:
    w_t = Pi(w_{t-1} + eta r_t),  z_{t+1} = Pi(w_t + eta r_t),
    sigma_{t+1} = z_{t+1}/||z_{t+1}||_1
Stable PRM+ (ibid., Alg. 1): plain PRM+ from w_0 = R0 1; whenever an
infoset's w_t <= R0 1 componentwise, reset w_t = R0 1 and drop its
prediction for the next step.
ExRM+ (ibid., Alg. 5; simultaneous updates only):
    w_t = Pi(z_{t-1} + eta r(z_{t-1})),  play sigma_t = w_t/||w_t||_1,
    z_t = Pi(z_{t-1} + eta r(w_t)).
IR-PRM+ (Zhang, Anagnostides & Sandholm 2025, Alg. 2) with the last
utility vector as prediction (their alternating IR-PCFR+):
    NextStrategy(m): if [rt]^+ = 0 at an infoset: m = 0, x = xt there;
      else find g with ||[rt + m - g 1]^+||_2 = ||rt||_2, r = rt + m - g 1,
      x = [r]^+/||[r]^+||_1
    ObserveUtility(u): g = u - m - <u - m, x> 1, rt = [r + g]^+,
      xt = rt/||rt||_1

Both projections are exact sort-based rules applied to all infosets of the
player at once (``np.lexsort`` by infoset, then ``np.*.reduceat``).
"""

from typing import Optional

import numpy as np

from .efg import EFG

__all__ = ["DiscountedRM", "SmoothPRM", "StablePRM", "ExRM", "IRPRM",
           "make_local", "hs_gamma", "project_l1_floor", "solve_l2_shift"]

def _iset_reduce(ufunc, pt, vals):
    """Per-infoset reduction over sequences 1..n_seq-1 (index 0 is the empty
    sequence and belongs to no infoset)."""
    return ufunc.reduceat(vals[1:], pt.iset_start - 1)


def _expand(pt, per_iset):
    """Broadcast a per-infoset array back to sequence length (0 at index 0)."""
    out = np.zeros(pt.n_seq)
    out[1:] = per_iset[pt.seq_iset[1:]]
    return out


def _segment_sort_desc(pt, y):
    """Sequences 1..n_seq-1 grouped by infoset (segments stay contiguous and
    in order) and sorted in decreasing value inside each segment. Returns the
    sorted values, the 1-based rank inside the segment and the segment start
    (in the shifted 0-based indexing of y[1:])."""
    iset = pt.seq_iset[1:]
    order = np.lexsort((-y[1:], iset))
    ys = y[1:][order]
    start = pt.iset_start - 1
    rank = np.arange(len(ys)) - start[iset] + 1
    return ys, rank, start


def _segment_cumsum(vals, start):
    cs = np.cumsum(vals)
    offset = np.concatenate(([0.0], cs[start[1:] - 1]))
    return cs - np.repeat(offset, np.diff(np.append(start, len(vals))))


def project_l1_floor(efg: EFG, i: int, y: np.ndarray, floor: float) -> np.ndarray:
    """Euclidean projection of y onto {R >= 0, ||R_I||_1 >= floor} per infoset.

    If [y]^+ already satisfies the floor the projection is [y]^+; otherwise the
    l1 constraint is active and the projection is [y - theta]^+ with theta
    solving sum_a [y_a - theta]^+ = floor (projection onto the scaled
    simplex, computed exactly by the sort-based rule: with y sorted
    decreasingly, rho = max{j : y_j - (sum_{k<=j} y_k - floor)/j > 0} and
    theta = (sum_{k<=rho} y_k - floor)/rho).
    """
    pt = efg.players[i]
    p = np.maximum(y, 0.0)
    p[0] = 0.0
    if pt.n_iset == 0:
        return p
    sums = _iset_reduce(np.add, pt, p)
    need = sums < floor
    if not need.any():
        return p
    ys, rank, start = _segment_sort_desc(pt, y)
    css = _segment_cumsum(ys, start)
    cond = ys - (css - floor) / rank > 0.0
    rho = np.add.reduceat(cond.astype(np.int64), start)
    theta = (css[start + rho - 1] - floor) / rho
    proj = np.maximum(y - _expand(pt, theta), 0.0)
    proj[0] = 0.0
    mask = _expand(pt, need.astype(float)) > 0.5
    return np.where(mask, proj, p)


def solve_l2_shift(efg: EFG, i: int, v: np.ndarray, target_sq: np.ndarray,
                   need: np.ndarray) -> np.ndarray:
    """Per infoset, the shift g with ||[v - g 1]^+||_2^2 = target_sq (target
    given per infoset, > 0 where ``need``). Returns g per infoset (0 where
    not needed).

    h(g) = sum_a [v_a - g]^+^2 is continuous and decreasing. With v sorted
    decreasingly and k active components, h(g) = k g^2 - 2 S_k g + Q_k on the
    piece v_{k+1} <= g < v_k; its root there is g_k = (S_k - sqrt(S_k^2 -
    k (Q_k - c^2)))/k, and exactly one k puts g_k inside its piece.
    """
    pt = efg.players[i]
    ys, rank, start = _segment_sort_desc(pt, v)
    s_k = _segment_cumsum(ys, start)
    q_k = _segment_cumsum(np.square(ys), start)
    iset = pt.seq_iset[1:]
    c_sq = target_sq[iset]
    disc = np.maximum(s_k * s_k - rank * (q_k - c_sq), 0.0)
    g_k = (s_k - np.sqrt(disc)) / rank
    upper = ys                                      # g_k < v_k
    end = np.append(start[1:], len(ys)) - 1         # last index of each segment
    is_last = np.arange(len(ys)) == end[iset]
    lower = np.where(is_last, -np.inf, np.append(ys[1:], -np.inf))   # g_k >= v_{k+1}
    valid = (g_k < upper) & (g_k >= lower) & need[iset]
    cand = np.where(valid, g_k, -np.inf)
    g = np.maximum.reduceat(cand, start)
    return np.where(need & np.isfinite(g), g, 0.0)


def hs_gamma(s: int, t_max: int, gamma0: float, slope: float) -> float:
    """HS_gamma schedule of Zhang et al. (2026), eq. 4: gamma_s = gamma0 - slope*s/n."""
    return gamma0 - slope * s / t_max


class DiscountedRM:
    """DCFR (``plus=False``), DCFR+ (``plus=True, predictive=False``) and
    PDCFR+ (``plus=True, predictive=True``); ``schedule='hs'`` replaces the
    constant alpha/beta by the HS-DCFR schedules."""

    def __init__(self, efg: EFG, i: int, alpha: float, beta: float,
                 plus: bool, predictive: bool, schedule: str, t_max: int):
        self.efg, self.i = efg, i
        self.alpha, self.beta = alpha, beta
        self.plus, self.predictive, self.schedule = plus, predictive, schedule
        self.t_max = t_max
        self.state = np.zeros(efg.players[i].n_seq)   # S_t = R_t (.) d_t

    def _ab(self, t: int):
        if self.schedule == "hs":
            return 1.0 + 3.0 * t / self.t_max, -1.0 - 2.0 * t / self.t_max
        return self.alpha, self.beta

    def step(self, ir: np.ndarray, v: np.ndarray, t: int) -> np.ndarray:
        a, b = self._ab(t)
        ta = float(t) ** a
        da = ta / (ta + 1.0)
        if self.plus:
            r = np.maximum(self.state + ir, 0.0)
            self.state = r * da
            if self.predictive:
                return self.efg.normalize(self.i, np.maximum(self.state + ir, 0.0))
            return self.efg.normalize(self.i, r)
        r = self.state + ir
        tb = float(t) ** b
        db = tb / (tb + 1.0)
        self.state = r * np.where(r > 0.0, da, db)
        return self.efg.normalize(self.i, np.maximum(r, 0.0))


class SmoothPRM:
    def __init__(self, efg: EFG, i: int, eta: float, floor: float):
        self.efg, self.i, self.eta, self.floor = efg, i, eta, floor
        self.w = floor * efg.uniform_sigma(i)
        self.w[0] = 0.0

    def step(self, ir: np.ndarray, v: np.ndarray, t: int) -> np.ndarray:
        self.w = project_l1_floor(self.efg, self.i, self.w + self.eta * ir, self.floor)
        z = project_l1_floor(self.efg, self.i, self.w + self.eta * ir, self.floor)
        return self.efg.normalize(self.i, z)


class StablePRM:
    def __init__(self, efg: EFG, i: int, eta: float, r0: float):
        self.efg, self.i, self.eta, self.r0 = efg, i, eta, r0
        pt = efg.players[i]
        self.w = np.full(pt.n_seq, r0)
        self.w[0] = 0.0

    def step(self, ir: np.ndarray, v: np.ndarray, t: int) -> np.ndarray:
        pt = self.efg.players[self.i]
        w = np.maximum(self.w + self.eta * ir, 0.0)
        w[0] = 0.0
        pred = ir
        if pt.n_iset:
            restart = _iset_reduce(np.maximum, pt, w) <= self.r0
            if restart.any():
                mask = _expand(pt, restart.astype(float)) > 0.5
                w = np.where(mask, self.r0, w)
                w[0] = 0.0
                pred = np.where(mask, 0.0, ir)
        self.w = w
        return self.efg.normalize(self.i, np.maximum(w + self.eta * pred, 0.0))


class ExRM:
    """Two-phase update; ``half`` uses the regret at the profile g(z), ``step``
    the regret at the played profile g(w)."""

    def __init__(self, efg: EFG, i: int, eta: float, floor: float):
        self.efg, self.i, self.eta, self.floor = efg, i, eta, floor
        self.z = floor * efg.uniform_sigma(i)
        self.z[0] = 0.0
        self.w = self.z.copy()

    def z_sigma(self) -> np.ndarray:
        return self.efg.normalize(self.i, self.z)

    def half(self, ir_z: np.ndarray) -> np.ndarray:
        self.w = project_l1_floor(self.efg, self.i, self.z + self.eta * ir_z, self.floor)
        return self.efg.normalize(self.i, self.w)

    def step(self, ir: np.ndarray, v: np.ndarray, t: int) -> Optional[np.ndarray]:
        self.z = project_l1_floor(self.efg, self.i, self.z + self.eta * ir, self.floor)
        return None   # the played strategy is produced by half() next iteration


class IRPRM:
    def __init__(self, efg: EFG, i: int):
        self.efg, self.i = efg, i
        pt = efg.players[i]
        self.rt = np.zeros(pt.n_seq)         # r~ (nonnegative)
        self.r = np.zeros(pt.n_seq)          # r after accepting the prediction
        self.m = np.zeros(pt.n_seq)          # effective prediction m^(t)
        self.xt = efg.uniform_sigma(i)       # x~
        self.x = self.xt.copy()              # strategy actually played

    def next_sigma(self, m: np.ndarray) -> np.ndarray:
        pt = self.efg.players[self.i]
        if pt.n_iset == 0:
            return self.xt.copy()
        norm_sq = _iset_reduce(np.add, pt, np.square(self.rt))
        need = norm_sq > 0.0
        vv = self.rt + m
        g = solve_l2_shift(self.efg, self.i, vv, norm_sq, need)
        r = vv - _expand(pt, g)
        r[0] = 0.0
        mask = _expand(pt, need.astype(float)) > 0.5
        self.m = np.where(mask, m, 0.0)
        self.m[0] = 0.0
        self.r = np.where(mask, r, self.rt)
        x_new = self.efg.normalize(self.i, np.maximum(r, 0.0))
        self.x = np.where(mask, x_new, self.xt)
        self.x[0] = 1.0
        return self.x

    def step(self, ir: np.ndarray, v: np.ndarray, t: int) -> np.ndarray:
        pt = self.efg.players[self.i]
        d = v - self.m
        if pt.n_iset:
            inner = _iset_reduce(np.add, pt, d * self.x)
            g = d - _expand(pt, inner)
        else:
            g = d.copy()
        g[0] = 0.0
        self.rt = np.maximum(self.r + g, 0.0)
        self.rt[0] = 0.0
        self.xt = self.efg.normalize(self.i, self.rt)
        return self.next_sigma(v)


def make_local(efg: EFG, i: int, cfg):
    """Instantiate the local minimizer of ``cfg.alg`` for player i (None for
    the three legacy algorithms handled inline by run_cfr)."""
    alg = cfg.alg
    if alg == "dcfr":
        return DiscountedRM(efg, i, cfg.disc_alpha, cfg.disc_beta, plus=False,
                            predictive=False, schedule=cfg.disc_schedule,
                            t_max=cfg.t_max)
    if alg == "dcfr+":
        return DiscountedRM(efg, i, cfg.disc_alpha, cfg.disc_beta, plus=True,
                            predictive=False, schedule=cfg.disc_schedule,
                            t_max=cfg.t_max)
    if alg == "pdcfr+":
        return DiscountedRM(efg, i, cfg.disc_alpha, cfg.disc_beta, plus=True,
                            predictive=True, schedule=cfg.disc_schedule,
                            t_max=cfg.t_max)
    if alg == "sprm+":
        return SmoothPRM(efg, i, cfg.step_eta, cfg.smooth_floor)
    if alg == "stprm+":
        return StablePRM(efg, i, cfg.step_eta, cfg.stable_r0)
    if alg == "exrm+":
        return ExRM(efg, i, cfg.step_eta, cfg.smooth_floor)
    if alg == "irprm+":
        return IRPRM(efg, i)
    return None
