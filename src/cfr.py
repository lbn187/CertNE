"""Simultaneous RM+ / PCFR+ / DPRM+ main loop over a sequence-form EFG.

DPRM+ ("decorrelation-predictive RM+", ours) feeds the online KL
decorrelation gradient through the bounded-prediction channel of predictive
regret matching:

    prediction(I, a) = ir(I, a) + clip(-lambda * g(I, a), +-B_t),
    B_t = clip_c0 * t^clip_alpha,   sigma^{t+1}(I) ~ [R + prediction]^+.

Average strategies, the empirical joint terminal distribution, and the KL
accumulators all use iteration weights w_t = t^weight_exp (0 = uniform,
1 = linear/CFR+, 2 = quadratic).

``update="alternating"`` follows the NPCFR paper protocol for the plus
family: players are updated one at a time within an iteration, each seeing
the already-updated strategies of earlier players. Averages/j_sum are still
accumulated once per iteration from the start-of-iteration profile, so the
certificate is exact for simultaneous runs but only approximate under
alternation (later players' regrets are measured against partially updated
opponents). Optional ``strict_regret_logging`` independently measures every
player against the start-of-round profile, without feeding those values back
into strategy updates, and provides an aligned alternating certificate.

Independently of alignment, ``d_seen``/``coal_seen`` accumulate, for each
player, the joint and opponent-coalition terminal reach of the profile its
regret was actually measured against. The profile-sequence certificate
(``certificate_seq``, metrics.certificate_seq_terms) is then sound for every
update scheme without the extra pass: the alignment mismatch becomes a
measured signed term instead of an assumption.
"""

import math
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import metrics as M
from .decorr import decorr_bias
from .efg import EFG
from .variants import hs_gamma, make_local

__all__ = ["CFRConfig", "run_cfr"]

# The first three are updated inline below; the rest are the arm-pool
# variants of src/variants.py (DCFR family, stabilised PRM+ family, IR-PRM+).
LEGACY_ALGS = ("rm+", "pcfr+", "dprm+")
VARIANT_ALGS = ("dcfr", "dcfr+", "pdcfr+", "sprm+", "stprm+", "exrm+", "irprm+")
ALGS = LEGACY_ALGS + VARIANT_ALGS
UPDATES = ("simultaneous", "alternating")
WEIGHT_SCHEDULES = ("power", "hs")
DISC_SCHEDULES = ("const", "hs")
RECORD_MODES = ("full", "none", "gap", "c0", "c1")


@dataclass(frozen=True)
class CFRConfig:
    alg: str = "dprm+"
    t_max: int = 1 << 14
    lam: float = 1e3
    clip_c0: float = 8.0
    clip_alpha: float = 0.25
    weight_exp: float = 2.0
    own_reach: str = "exclude"   # decorr bias: counterfactual vs true gradient
    center_bias: bool = False
    update: str = "simultaneous"
    eval_every: int = 1024
    strict_regret_logging: bool = False
    checkpoints: Optional[Tuple[int, ...]] = None
    # What a checkpoint records, and therefore what it pays for. "full" is the
    # released configuration; the others exist so the per-checkpoint cost of a
    # certificate can be measured end to end against training with no
    # checkpoint work at all.
    #   full : every released quantity (exact gap, C^0, C^1, tight, ...)
    #   none : nothing but the iteration index
    #   gap  : exact NashGap only (one best-response pass per player)
    #   c0   : C^0 terms only (no backward pass)
    #   c1   : C^1 terms (one signed backward pass per player)
    record_mode: str = "full"
    # Whether the certificate accumulators are maintained: unclipped regret
    # sums, the profile each player's regret actually saw, and the joint
    # terminal distribution. False runs the bare solver; record_mode must
    # then be "none" or "gap", the modes that do not read them.
    certificate_state: bool = True
    init_mode: str = "uniform"
    init_seed: int = 0
    dirichlet_alpha: float = 1.0
    # ---- arm-pool variants (defaults leave the three legacy algorithms and
    # their iteration weights w_t = t^weight_exp bit-for-bit unchanged) ----
    # DCFR-family regret discounting: positive regrets by t^a/(t^a+1),
    # negative by t^b/(t^b+1); "hs" uses a_t = 1 + 3t/T, b_t = -1 - 2t/T.
    disc_alpha: float = 1.5
    disc_beta: float = 0.0
    disc_schedule: str = "const"
    # Iteration weights of every accumulator (averages, joint distribution,
    # certificate sums). "power": w_t = t^weight_exp. "hs": gamma_t =
    # hs_gamma0 - hs_gamma_slope * t/T and w_{t+1} = w_t ((t+1)/t)^gamma_t,
    # i.e. DCFR's (t/(t+1))^gamma averaging with a time-varying gamma.
    weight_schedule: str = "power"
    hs_gamma0: float = 30.0
    hs_gamma_slope: float = 5.0
    # Stabilised PRM+ family: step size and the l1 floor of the chopped
    # decision set (Smooth PRM+, ExRM+) / the restart level (Stable PRM+).
    step_eta: float = 1.0
    smooth_floor: float = 1.0
    stable_r0: float = 1.0

    def __post_init__(self) -> None:
        """Validate iteration recording and initialization settings."""
        if self.t_max < 1:
            raise ValueError("t_max must be positive")
        if self.eval_every < 1:
            raise ValueError("eval_every must be positive")
        if self.alg not in ALGS:
            raise ValueError(f"unknown alg {self.alg!r}, expected one of {ALGS}")
        if self.update not in UPDATES:
            raise ValueError(f"unknown update {self.update!r}, expected {UPDATES}")
        if self.alg == "exrm+" and self.update != "simultaneous":
            raise ValueError("exrm+ is defined for simultaneous updates only")
        if self.weight_schedule not in WEIGHT_SCHEDULES:
            raise ValueError(f"unknown weight_schedule {self.weight_schedule!r}")
        if self.disc_schedule not in DISC_SCHEDULES:
            raise ValueError(f"unknown disc_schedule {self.disc_schedule!r}")
        if self.weight_schedule == "hs":
            # w_T ~ exp(gamma0 log T); keep it far below the float64 overflow
            if self.hs_gamma0 * math.log(self.t_max) > 600.0:
                raise ValueError("hs weight schedule overflows float64 for this "
                                 "gamma0 / t_max")
        for name in ("step_eta", "smooth_floor", "stable_r0"):
            val = getattr(self, name)
            if not np.isfinite(val) or val <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.checkpoints is not None:
            points = self.checkpoints
            if not isinstance(points, tuple) or not points or any(
                    type(t) is not int for t in points):
                raise ValueError("checkpoints must be a non-empty tuple of integers")
            if any(t < 1 or t > self.t_max for t in points):
                raise ValueError("checkpoints must lie in [1, t_max]")
            if any(a >= b for a, b in zip(points, points[1:])):
                raise ValueError("checkpoints must be strictly increasing")
            if points[-1] != self.t_max:
                raise ValueError("checkpoints must explicitly include t_max")
        if self.record_mode not in RECORD_MODES:
            raise ValueError(f"unknown record_mode {self.record_mode!r}")
        if not self.certificate_state and self.record_mode not in ("none", "gap"):
            raise ValueError(
                "certificate_state=False needs record_mode 'none' or 'gap'")
        if self.init_mode not in ("uniform", "random_dirichlet"):
            raise ValueError(f"unknown init_mode {self.init_mode!r}")
        if type(self.init_seed) is not int or self.init_seed < 0:
            raise ValueError("init_seed must be a nonnegative integer")
        if not np.isfinite(self.dirichlet_alpha) or self.dirichlet_alpha <= 0.0:
            raise ValueError("dirichlet_alpha must be positive and finite")


def _initial_sigma(efg: EFG, cfg: CFRConfig) -> List[np.ndarray]:
    """Construct a reproducible initial behavioral profile."""
    if cfg.init_mode == "uniform":
        return [efg.uniform_sigma(i) for i in range(efg.n_players)]
    rng = np.random.default_rng(cfg.init_seed)
    sigma: List[np.ndarray] = []
    for i, pt in enumerate(efg.players):
        values = np.zeros(pt.n_seq)
        values[0] = 1.0
        for start, end in zip(pt.iset_start, pt.iset_end):
            values[start:end] = rng.dirichlet(
                np.full(end - start, cfg.dirichlet_alpha))
        sigma.append(values)
    return sigma


def run_cfr(efg: EFG, cfg: CFRConfig) -> Dict:
    """Run the configured algorithm; returns final averages + metric history."""
    if cfg.alg not in ALGS:
        raise ValueError(f"unknown alg {cfg.alg!r}, expected one of {ALGS}")
    if cfg.update not in UPDATES:
        raise ValueError(f"unknown update {cfg.update!r}, expected {UPDATES}")
    n = efg.n_players
    trees = efg.players
    regret = [np.zeros(pt.n_seq) for pt in trees]       # RM+ clipped regrets
    regret_cum = ([np.zeros(pt.n_seq) for pt in trees]  # unclipped, certificate
                  if cfg.certificate_state else None)
    strict_regret_cum = ([np.zeros(pt.n_seq) for pt in trees]
                         if cfg.strict_regret_logging else None)
    sigma = _initial_sigma(efg, cfg)
    avg = [np.zeros(pt.n_seq) for pt in trees]          # weighted realization sums
    j_sum = np.zeros(efg.n_terminals)                   # weighted joint terminal dist
    # weighted opponent-coalition reach per player (Prop C' D_{-i}; n>=3 only,
    # for n=2 the coalition term is identically zero and cert_tight covers it)
    coal_sum = ([np.zeros(efg.n_terminals) for _ in range(n)]
                if n >= 3 and cfg.certificate_state else None)
    # profile actually seen by each player's regret (chance-weighted):
    # joint reach and opponent-coalition reach at the moment of measurement
    d_seen = ([np.zeros(efg.n_terminals) for _ in range(n)]
              if cfg.certificate_state else None)
    coal_seen = ([np.zeros(efg.n_terminals) for _ in range(n)]
                 if cfg.certificate_state else None)
    w_sum = 0.0
    history: Dict[str, List] = {
        "t": [], "nash_gap": [], "player_gaps": [], "corr_kl": [],
        "certificate": [], "avg_regret_bound": [],
    }
    selected_snapshot = None
    checkpoint_set = set(cfg.checkpoints) if cfg.checkpoints is not None else None
    # arm-pool variants: one local minimizer per player; None for legacy algs
    local = [make_local(efg, i, cfg) for i in range(n)]
    variant = local[0] is not None
    log_w = 0.0   # log w_t under the "hs" weight schedule (w_1 = 1)
    t_start = time.time()
    for t in range(cfg.t_max):
        if cfg.alg == "exrm+":
            # extragradient half-step: regret at the profile g(z), then play g(w)
            sig_z = [local[i].z_sigma() for i in range(n)]
            xs_z = [efg.realization(i, sig_z[i]) for i in range(n)]
            for i in range(n):
                term_w = efg.opp_reach(i, xs_z) * efg.term_util[i]
                v_z = efg.backward_sum(i, sig_z[i], term_w)
                sigma[i] = local[i].half(efg.local_regret(i, sig_z[i], v_z))
        xs = [efg.realization(i, sigma[i]) for i in range(n)]
        if cfg.weight_schedule == "power":
            w = float(t + 1) ** cfg.weight_exp
        else:
            w = math.exp(log_w)
            log_w += (hs_gamma(t + 1, cfg.t_max, cfg.hs_gamma0, cfg.hs_gamma_slope)
                      * math.log((t + 2) / (t + 1)))
        j_sum += w * efg.joint_reach(xs)
        for i in range(n):
            avg[i] += w * xs[i]
        if coal_sum is not None:
            reaches = [xs[j][efg.term_seq[j]] for j in range(n)]
            for i in range(n):
                prod = np.ones(efg.n_terminals)
                for j in range(n):
                    if j != i:
                        prod *= reaches[j]
                coal_sum[i] += w * prod
        w_sum += w

        if strict_regret_cum is not None:
            for i in range(n):
                term_w = efg.opp_reach(i, xs) * efg.term_util[i]
                v = efg.backward_sum(i, sigma[i], term_w)
                strict_regret_cum[i] += w * efg.local_regret(i, sigma[i], v)

        bound = cfg.clip_c0 * float(t + 1) ** cfg.clip_alpha
        if variant:
            # certificate accumulators exactly as below; only the strategy
            # update is delegated to the local minimizer (own ir and v only)
            new_sigma = list(sigma)
            for i in range(n):
                if cfg.update == "alternating" and i > 0:
                    xs = [efg.realization(j, sigma[j]) for j in range(n)]
                opp = efg.opp_reach(i, xs)
                if d_seen is not None:
                    coal_seen[i] += w * opp
                    d_seen[i] += w * opp * xs[i][efg.term_seq[i]]
                term_w = opp * efg.term_util[i]
                v = efg.backward_sum(i, sigma[i], term_w)
                ir = efg.local_regret(i, sigma[i], v)
                if regret_cum is not None:
                    regret_cum[i] += w * ir
                nxt = local[i].step(ir, v, t + 1)
                if nxt is not None:
                    if cfg.update == "alternating":
                        sigma[i] = nxt
                    else:
                        new_sigma[i] = nxt
            if cfg.update == "simultaneous":
                sigma = new_sigma
        elif cfg.update == "simultaneous":
            irs = []
            for i in range(n):
                opp = efg.opp_reach(i, xs)
                if d_seen is not None:
                    coal_seen[i] += w * opp
                    d_seen[i] += w * opp * xs[i][efg.term_seq[i]]
                term_w = opp * efg.term_util[i]
                v = efg.backward_sum(i, sigma[i], term_w)
                ir = efg.local_regret(i, sigma[i], v)
                regret[i] = np.maximum(regret[i] + ir, 0.0)
                # weighted, so the certificate matches the weighted avg profile
                if regret_cum is not None:
                    regret_cum[i] += w * ir
                irs.append(ir)
            if cfg.alg == "rm+":
                preds = [np.zeros_like(ir) for ir in irs]
            elif cfg.alg == "pcfr+":
                preds = irs
            else:  # dprm+
                grads = decorr_bias(efg, xs, sigma, j_sum, avg, w_sum, w,
                                    own_reach=cfg.own_reach,
                                    center=cfg.center_bias)
                preds = [irs[i] + np.clip(-cfg.lam * grads[i], -bound, bound)
                         for i in range(n)]
            sigma = [efg.normalize(i, np.maximum(regret[i] + preds[i], 0.0))
                     for i in range(n)]
        else:  # alternating: each player sees earlier players' updated sigma
            for i in range(n):
                if i > 0:
                    xs = [efg.realization(j, sigma[j]) for j in range(n)]
                opp = efg.opp_reach(i, xs)
                if d_seen is not None:
                    coal_seen[i] += w * opp
                    d_seen[i] += w * opp * xs[i][efg.term_seq[i]]
                term_w = opp * efg.term_util[i]
                v = efg.backward_sum(i, sigma[i], term_w)
                ir = efg.local_regret(i, sigma[i], v)
                regret[i] = np.maximum(regret[i] + ir, 0.0)
                if regret_cum is not None:
                    regret_cum[i] += w * ir
                if cfg.alg == "rm+":
                    pred = np.zeros_like(ir)
                elif cfg.alg == "pcfr+":
                    pred = ir
                else:  # dprm+
                    grads = decorr_bias(efg, xs, sigma, j_sum, avg, w_sum, w,
                                        own_reach=cfg.own_reach,
                                        center=cfg.center_bias)
                    pred = ir + np.clip(-cfg.lam * grads[i], -bound, bound)
                sigma[i] = efg.normalize(
                    i, np.maximum(regret[i] + pred, 0.0))

        should_record = ((t + 1) in checkpoint_set if checkpoint_set is not None
                         else (t + 1) % cfg.eval_every == 0 or t + 1 == cfg.t_max)
        if should_record:
            _record(efg, history, t + 1, avg, j_sum, w_sum, regret_cum,
                    coal_sum, strict_regret_cum, d_seen, coal_seen,
                    mode=cfg.record_mode)
            score_key = _snapshot_score_key(n, strict_regret_cum is not None,
                                            cfg.record_mode)
            if score_key is not None:
                score_status = _snapshot_score_status(cfg)
                score = history[score_key][-1]
                if selected_snapshot is None or score < selected_snapshot["score"]:
                    selected_snapshot = {
                        "xs_bar": [(values / w_sum).copy() for values in avg],
                        "t": t + 1,
                        "score": score,
                        "score_key": score_key,
                        "status": score_status,
                    }

    xs_bar = [avg[i] / w_sum for i in range(n)]
    # trajectory-minimum NashGap over checkpoints (the NPCFR paper's metric);
    # -1 when record_mode never evaluated the exact gap
    min_idx = (int(np.argmin(history["nash_gap"]))
               if history["nash_gap"] else -1)
    result = {
        "config": asdict(cfg),
        "xs_bar": xs_bar,
        "p_bar": j_sum / w_sum,
        "regret_cum": regret_cum,
        "history": history,
        "final": {k: v[-1] for k, v in history.items() if v},
        "min_nash_gap": history["nash_gap"][min_idx] if min_idx >= 0 else None,
        "min_nash_gap_t": history["t"][min_idx] if min_idx >= 0 else None,
        "selected_snapshot": selected_snapshot,
        "wall_seconds": time.time() - t_start,
    }
    if strict_regret_cum is not None:
        result["strict_regret_cum"] = strict_regret_cum
    return result


def _snapshot_score_key(n_players: int, strict: bool,
                        mode: str = "full") -> "str | None":
    """Return the score used for snapshot selection, or None if the mode
    records no score."""
    if mode == "none":
        return None
    if mode == "gap":
        return "nash_gap"
    if mode == "c0":
        return "certificate_seq_l1"
    if mode == "c1":
        return "certificate_seq"
    base = "certificate_tight" if n_players == 2 else "certificate_coal"
    return f"{base}_strict" if strict else base


def _snapshot_score_status(cfg: CFRConfig) -> str:
    """Classify the selected score as formal or heuristic.

    Simultaneous regret is aligned with the averaged profile. Alternating
    regret has the same guarantee only when start-of-round strict logging is
    enabled; its legacy partially updated-opponent score remains diagnostic.
    """
    if cfg.update == "simultaneous" or cfg.strict_regret_logging:
        return "formal"
    return "heuristic"


def _record(efg: EFG, history: Dict[str, List], t: int,
            avg: List[np.ndarray], j_sum: np.ndarray, w_sum: float,
            regret_cum: List[np.ndarray],
            coal_sum: "List[np.ndarray] | None" = None,
            strict_regret_cum: "List[np.ndarray] | None" = None,
            d_seen: "List[np.ndarray] | None" = None,
            coal_seen: "List[np.ndarray] | None" = None,
            mode: str = "full") -> None:
    n = efg.n_players
    xs_bar = [avg[i] / w_sum for i in range(n)]
    history["t"].append(t)
    if mode == "none":
        return
    if mode in ("gap", "c0", "c1"):
        # deployment cost of one arm: only what that arm scores with
        if mode == "gap":
            gap, gaps = M.nash_gap(efg, xs_bar)
            history["nash_gap"].append(gap)
            history["player_gaps"].append(gaps)
            return
        reg_bounds = [M.cfr_regret_bound(efg, i, regret_cum[i]) / w_sum
                      for i in range(n)]
        terms = M.certificate_seq_terms(
            efg, xs_bar, [d / w_sum for d in d_seen],
            [c / w_sum for c in coal_seen], reg_bounds,
            zero_br=(mode == "c0"))
        history.setdefault("certificate_seq_terms", []).append(terms)
        history.setdefault("certificate_seq", []).append(
            max(r + d + l for r, d, l, _, _ in terms))
        history.setdefault("certificate_seq_l1", []).append(
            max(r + d + l1 for r, d, _, l1, _ in terms))
        return
    gap, gaps = M.nash_gap(efg, xs_bar)
    kl = M.corr_kl(efg, j_sum / w_sum, xs_bar)
    # regret_cum is w_t-weighted, so normalize by w_sum (uniform: w_sum = t)
    reg_bounds = [M.cfr_regret_bound(efg, i, regret_cum[i]) / w_sum
                  for i in range(n)]
    history["nash_gap"].append(gap)
    history["player_gaps"].append(gaps)
    history["corr_kl"].append(kl)
    history["avg_regret_bound"].append(reg_bounds)
    history["certificate"].append(M.certificate(efg, reg_bounds, kl))
    if d_seen is not None:
        # sound for any update order: regret, value and comparator terms all
        # refer to the profile sequence each player's regret actually saw
        terms = M.certificate_seq_terms(
            efg, xs_bar, [d / w_sum for d in d_seen],
            [c / w_sum for c in coal_seen], reg_bounds)
        history.setdefault("certificate_seq_terms", []).append(terms)
        history.setdefault("certificate_seq", []).append(
            max(r + d + l for r, d, l, _, _ in terms))
        history.setdefault("certificate_seq_l1", []).append(
            max(r + d + l1 for r, d, _, l1, _ in terms))
        history.setdefault("certificate_seq_md", []).append(
            max(r + d + md for r, d, _, _, md in terms))
    strict_bounds = None
    if strict_regret_cum is not None:
        strict_bounds = [
            M.cfr_regret_bound(efg, i, strict_regret_cum[i]) / w_sum
            for i in range(n)
        ]
        history.setdefault("avg_regret_bound_strict", []).append(strict_bounds)
    if n == 2:  # Prop C-tight is only a sound bound for two players
        history.setdefault("certificate_tight", []).append(
            M.certificate_exact2p(efg, xs_bar, j_sum / w_sum, reg_bounds))
        if strict_bounds is not None:
            history.setdefault("certificate_tight_strict", []).append(
                M.certificate_exact2p(
                    efg, xs_bar, j_sum / w_sum, strict_bounds))
    elif coal_sum is not None:  # Prop C' with the coalition term D_{-i}
        coal_bar = [c / w_sum for c in coal_sum]
        history.setdefault("certificate_coal", []).append(
            M.certificate_exact(efg, xs_bar, j_sum / w_sum, coal_bar,
                                reg_bounds))
        # analysis-tier refinement: exact LBR comparator instead of M*D_{-i}
        # (one backward pass per player; not a zero-BR deployment signal)
        history.setdefault("certificate_coal_lbr", []).append(
            M.certificate_exact_lbr(efg, xs_bar, j_sum / w_sum, coal_bar,
                                    reg_bounds))
        if strict_bounds is not None:
            history.setdefault("certificate_coal_strict", []).append(
                M.certificate_exact(efg, xs_bar, j_sum / w_sum, coal_bar,
                                    strict_bounds))
            history.setdefault("certificate_coal_lbr_strict", []).append(
                M.certificate_exact_lbr(efg, xs_bar, j_sum / w_sum, coal_bar,
                                        strict_bounds))
