"""Evaluation metrics: NashGap (exact BR), correlation KL, runtime certificate."""

import math
from typing import List, Tuple

import numpy as np

from .efg import EFG

__all__ = ["utilities", "nash_gap", "corr_kl", "cfr_regret_bound",
           "certificate", "certificate_exact2p", "certificate_exact",
           "certificate_exact_lbr", "certificate_seq_terms"]

_KL_EPS = 1e-15


def utilities(efg: EFG, xs: List[np.ndarray]) -> List[float]:
    """Expected utility of each player under the profile of realization plans."""
    joint = efg.joint_reach(xs)
    return [float(joint @ efg.term_util[i]) for i in range(efg.n_players)]


def nash_gap(efg: EFG, xs: List[np.ndarray]) -> Tuple[float, List[float]]:
    """max_i (best-response value - current value), plus per-player gaps."""
    us = utilities(efg, xs)
    gaps = []
    for i in range(efg.n_players):
        term_w = efg.opp_reach(i, xs) * efg.term_util[i]
        gaps.append(efg.best_response_value(i, term_w) - us[i])
    return max(gaps), gaps


def corr_kl(efg: EFG, p_bar: np.ndarray, xs_bar: List[np.ndarray]) -> float:
    """KL(empirical joint terminal dist || product of averaged marginals)."""
    q = efg.term_chance.copy()
    for i in range(efg.n_players):
        q = q * xs_bar[i][efg.term_seq[i]]
    mask = p_bar > _KL_EPS
    return float(np.sum(
        p_bar[mask] * np.log(p_bar[mask] / (q[mask] + _KL_EPS))
    ))


def cfr_regret_bound(efg: EFG, i: int, r_cum: np.ndarray) -> float:
    """CFR bound on player i's cumulative external regret:
    sum over infosets of [max_a R_cum(I,a)]^+ (measurable online)."""
    pt = efg.players[i]
    if pt.n_iset == 0:
        return 0.0
    best = np.maximum.reduceat(r_cum, pt.iset_start)
    return float(np.maximum(best, 0.0).sum())


def certificate(efg: EFG, avg_regret_bounds: List[float], kl: float) -> float:
    """Best-response-free NashGap upper bound (Thm-1 style):
    max_i avg-regret bound + M * sqrt(2 * eps), M = max |u|.
    """
    m_u = float(np.abs(efg.term_util).max())
    return max(avg_regret_bounds) + m_u * math.sqrt(max(2.0 * kl, 0.0))


def certificate_exact2p(efg: EFG, xs_bar: List[np.ndarray],
                        p_bar: np.ndarray,
                        avg_regret_bounds: List[float]) -> float:
    """Exact BR-free NashGap upper bound, VALID FOR n=2 ONLY (Prop C-tight,
    notes/THEORY_PROPC.md §2b): for two players the best-response term is
    linear in the single opponent's averaged plan, so

        gap_i <= R_bar_i + ( <d_bar, u_i> - u_i(sigma_bar) ),

    with no Pinsker/KL slack; both correction terms are online-measurable
    (weighted empirical joint p_bar and the averaged profile xs_bar). The
    signed correction can be negative. For n >= 3 this MISSES the opponent-
    coalition correlation term D_{-i} and is not a sound bound.
    """
    if efg.n_players != 2:
        raise ValueError("certificate_exact2p is only valid for 2 players")
    mu_bar = efg.joint_reach(xs_bar)
    diff = p_bar - mu_bar
    return max(avg_regret_bounds[i] + float(diff @ efg.term_util[i])
               for i in range(2))


def certificate_exact(efg: EFG, xs_bar: List[np.ndarray],
                      p_bar: np.ndarray, coal_bar: List[np.ndarray],
                      avg_regret_bounds: List[float]) -> float:
    """General-n exact BR-free NashGap upper bound (Prop C', THEORY_PROPC §3):

        gap_i <= R_bar_i + (<d_bar,u_i> - u_i(sigma_bar)) + M * D_{-i},
        D_{-i} = sum_z d_c(z) |coal_bar_i(z) - prod_{j!=i} x_bar_j(z)|,

    where coal_bar_i(z) = (1/W) sum_t w_t prod_{j!=i} x_j^t(z) is the
    weighted empirical opponent-coalition reach (accumulated online in
    run_cfr). For n=2, D_{-i} = 0 identically and this reduces to
    certificate_exact2p. The coalition term is what the KL certificate
    cannot see at terminals player i never reaches.
    """
    n = efg.n_players
    m_u = float(np.abs(efg.term_util).max())
    mu_bar = efg.joint_reach(xs_bar)
    diff = p_bar - mu_bar
    vals = []
    for i in range(n):
        prod = np.ones(efg.n_terminals)
        for j in range(n):
            if j != i:
                prod *= xs_bar[j][efg.term_seq[j]]
        d_i = float(np.abs(coal_bar[i] - prod) @ efg.term_chance)
        vals.append(avg_regret_bounds[i] + float(diff @ efg.term_util[i])
                    + m_u * d_i)
    return max(vals)


def certificate_exact_lbr(efg: EFG, xs_bar: List[np.ndarray],
                          p_bar: np.ndarray, coal_bar: List[np.ndarray],
                          avg_regret_bounds: List[float]) -> float:
    """General-n NashGap upper bound with the EXACT coalition comparator:

        gap_i <= R_bar_i + (<d_bar,u_i> - u_i(sigma_bar)) + L_{-i},
        L_{-i} = max_{x'} <x', w_i>,
        w_i(z) = d_c(z) u_i(z) (prod_{j!=i} x_bar_j(z) - coal_bar_i(z)),

    where the max is the sequence-form best response against the SIGNED
    imbalance weight w_i (one backward pass per player), so
    L_{-i} <= M * D_{-i} always and L_{-i} can be negative. This is an
    analysis-tier quantity: the pass costs the same as one exact gap
    evaluation, so it does not replace the zero-BR certificate_exact;
    its role is to attribute the coalition certificate's slack to the
    M*D_{-i} relaxation versus the regret/alignment terms.
    """
    n = efg.n_players
    mu_bar = efg.joint_reach(xs_bar)
    diff = p_bar - mu_bar
    vals = []
    for i in range(n):
        prod = np.ones(efg.n_terminals)
        for j in range(n):
            if j != i:
                prod *= xs_bar[j][efg.term_seq[j]]
        w_i = efg.term_chance * efg.term_util[i] * (prod - coal_bar[i])
        l_i = efg.best_response_value(i, w_i)
        vals.append(avg_regret_bounds[i] + float(diff @ efg.term_util[i])
                    + l_i)
    return max(vals)


def certificate_seq_terms(efg: EFG, xs_bar: List[np.ndarray],
                          d_seen: List[np.ndarray],
                          coal_seen: List[np.ndarray],
                          avg_regret_bounds: List[float],
                          zero_br: bool = False) -> List[List[float]]:
    """Per-player terms of the profile-sequence certificate (any n, any
    update order). Player i's regret accumulator is measured against the
    opponent profiles y_{-i}^t it actually saw (start-of-round under
    simultaneous updates, partially updated under alternation), and

        gap_i <= Rhat_i + Delta_i^seen + L_i^seen,   L_i^seen <= M * D_i^seen,
        Delta_i^seen = <d_seen_i, u_i> - u_i(xs_bar),
        L_i^seen = max_{x'} <x', w_i>,
    (with zero_br=True the backward pass is skipped and the reported L is the
    zero-best-response relaxation ||w_i||_1 instead, which is what C^0 pays),
        w_i(z) = u_i(z) (p_c(z) prod_{j!=i} xbar_j(z) - coal_seen_i(z)),
        D_i^seen = sum_z |p_c(z) prod_{j!=i} xbar_j(z) - coal_seen_i(z)|,

    where d_seen_i = (1/W) sum_t w_t p_c prod_j y_j^t (with y_i^t = x_i^t)
    and coal_seen_i = (1/W) sum_t w_t p_c prod_{j!=i} y_j^t, both already
    chance-weighted. Returns [Rhat_i, Delta_i^seen, L_i^seen, ||w_i||_1,
    M*D_i^seen] per player; the exact certificate is max_i of the first
    three summed (one signed backward pass per player per checkpoint), and
    L_i^seen <= ||w_i||_1 <= M*D_i^seen are its two zero-BR relaxations.
    For simultaneous updates this reduces to certificate_exact_lbr /
    certificate_exact; for n=2 simultaneous, w_i = 0.
    """
    n = efg.n_players
    m_u = float(np.abs(efg.term_util).max())
    mu_bar = efg.joint_reach(xs_bar)
    terms = []
    for i in range(n):
        prod = efg.opp_reach(i, xs_bar)
        imbalance = prod - coal_seen[i]
        w_i = efg.term_util[i] * imbalance
        l1_i = float(np.abs(w_i).sum())
        l_i = l1_i if zero_br else efg.best_response_value(i, w_i)
        md_i = m_u * float(np.abs(imbalance).sum())
        delta_i = float((d_seen[i] - mu_bar) @ efg.term_util[i])
        terms.append([avg_regret_bounds[i], delta_i, l_i, l1_i, md_i])
    return terms
