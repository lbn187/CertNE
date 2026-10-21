"""Online decorrelation gradient — the training-free replacement for NPCFR.

The correlation objective is the KL of the (weight-averaged) empirical joint
terminal distribution against the product of the players' averaged marginal
reach distributions (the epsilon-EFM measure of arXiv:2504.18868):

    p(z) = (J(z) + w * c(z) prod_i d_i(z)) / (W + w)
    q(z) = c(z) * prod_i (M_i(z) + w * d_i(z)) / (W + w)
    L    = sum_z p(z) log(p(z) / q(z))

CONVENTION (deliberate, byte-compatible with the NFG pilot's kl_grads): the
accumulators J, M_i passed in by run_cfr ALREADY include the current iterate
t; this module then adds the current iterate AGAIN with weight w as a proxy
for the (unknown) next iterate. The proxy therefore carries total weight 2w
relative to W. Changing this convention rescales the effective lambda —
do not "fix" it without re-tuning (see review finding P1-1).
The gradient w.r.t. player i's terminal reach is

    dL/dd_i(z) = (w/(W+w)) * [ c(z) prod_{j!=i} d_j(z) * (log(p/q) + 1)
                               - p(z) (W+w) / (M_i(z) + w d_i(z)) ]

Everything returned here is scaled by (W+w)/w (absorbed into lambda), and
chained through the sequence tree so ``bias[s]`` lives in the same
per-infoset space as counterfactual regrets.
"""

from typing import List

import numpy as np

from .efg import EFG

__all__ = ["decorr_loss", "decorr_bias", "EPS"]

EPS = 1e-12


def _terminal_quantities(efg: EFG, xs: List[np.ndarray], j_sum: np.ndarray,
                         avg: List[np.ndarray], w_sum: float, w: float):
    d = [xs[i][efg.term_seq[i]] for i in range(efg.n_players)]
    joint = efg.joint_reach(xs)
    p = (j_sum + w * joint) / (w_sum + w)
    q = efg.term_chance.copy()
    m_factors = []
    for i in range(efg.n_players):
        m_i = avg[i][efg.term_seq[i]] + w * d[i]
        m_factors.append(m_i)
        q = q * (m_i / (w_sum + w))
    return d, p, q, m_factors


def decorr_loss(efg: EFG, xs: List[np.ndarray], j_sum: np.ndarray,
                avg: List[np.ndarray], w_sum: float, w: float) -> float:
    """Exact scalar L (used by the finite-difference gradient test)."""
    _, p, q, _ = _terminal_quantities(efg, xs, j_sum, avg, w_sum, w)
    return float(np.sum(p * np.log((p + EPS) / (q + EPS))))


def decorr_bias(efg: EFG, xs: List[np.ndarray], sigma: List[np.ndarray],
                j_sum: np.ndarray, avg: List[np.ndarray], w_sum: float,
                w: float, own_reach: str = "exclude",
                center: bool = False) -> List[np.ndarray]:
    """Per-player, per-sequence KL gradient (scaled by (W+w)/w).

    own_reach='include' returns the true dL/dsigma(I,a); 'exclude' drops the
    player's own reach to the infoset (counterfactual convention, matching
    the space CFR regrets live in). center=True subtracts the per-infoset
    mean of the gradient.
    """
    _, p, q, m_factors = _terminal_quantities(efg, xs, j_sum, avg, w_sum, w)
    ratio = np.log((p + EPS) / (q + EPS)) + 1.0
    out: List[np.ndarray] = []
    for i in range(efg.n_players):
        opp = efg.opp_reach(i, xs)
        g_term = opp * ratio - p * (w_sum + w) / (m_factors[i] + EPS)
        h = efg.backward_sum(i, sigma[i], g_term)
        pt = efg.players[i]
        if own_reach == "include":
            g = np.zeros_like(h)
            g[1:] = xs[i][pt.seq_parent[1:]] * h[1:]
        elif own_reach == "exclude":
            g = h.copy()
            g[0] = 0.0
        else:
            raise ValueError(own_reach)
        if center and pt.n_iset:
            sums = np.add.reduceat(g, pt.iset_start)
            counts = (pt.iset_end - pt.iset_start).astype(float)
            g[1:] -= (sums / counts)[pt.seq_iset[1:]]
        out.append(g)
    return out
