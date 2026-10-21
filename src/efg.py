"""Sequence-form extensive-form game engine.

A generic game definition (see src/games/) is flattened once into terminal
arrays plus one sequence tree per player. All per-iteration CFR operations
(realization plans, backward value sums, best responses, per-infoset
normalization) are vectorized over sequences, grouped by depth, so a full
pass costs O(max_depth) numpy calls per player.

Sequence indexing convention (per player): sequence 0 is the empty sequence;
each infoset owns a contiguous block of sequence ids, one per action, and
infoset ids are ordered so that ``iset_start`` is strictly increasing.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["PlayerTree", "EFG", "build_efg"]


@dataclass
class PlayerTree:
    """Sequence tree of one player (arrays as described in the module docstring)."""

    n_seq: int
    seq_parent: np.ndarray      # [n_seq] parent sequence of the owning infoset
    seq_iset: np.ndarray        # [n_seq] owning infoset id (-1 for sequence 0)
    iset_start: np.ndarray      # [n_iset] first sequence id of the infoset
    iset_end: np.ndarray        # [n_iset] one past the last sequence id
    iset_parent_seq: np.ndarray  # [n_iset]
    iset_depth: np.ndarray      # [n_iset] number of own decisions on the path
    level_seq: List[np.ndarray]     # per depth d>=1: sequence ids (ascending)
    level_iset: List[np.ndarray]    # per depth: infoset ids (same order)
    level_offsets: List[np.ndarray]  # per depth: reduceat offsets into level_seq
    iset_keys: List[Any]

    @property
    def n_iset(self) -> int:
        return len(self.iset_start)


class _PlayerBuilder:
    """Accumulates infosets/sequences during the tree walk."""

    def __init__(self) -> None:
        self.key2id: Dict[Any, int] = {}
        self.key2actions: Dict[Any, tuple] = {}
        self.iset_start: List[int] = []
        self.iset_end: List[int] = []
        self.iset_parent_seq: List[int] = []
        self.iset_depth: List[int] = []
        self.seq_parent: List[int] = [0]
        self.seq_iset: List[int] = [-1]
        self.keys: List[Any] = []

    def infoset(self, key: Any, parent_seq: int, depth: int,
                actions: tuple) -> int:
        existing = self.key2id.get(key)
        if existing is not None:
            if self.iset_parent_seq[existing] != parent_seq:
                raise ValueError(f"perfect recall violated at infoset {key!r}")
            if self.iset_depth[existing] != depth:
                raise ValueError(f"inconsistent depth at infoset {key!r}")
            if self.key2actions[key] != actions:
                raise ValueError(f"inconsistent actions at infoset {key!r}")
            return existing
        n_actions = len(actions)
        self.key2actions[key] = actions
        iset = len(self.iset_start)
        self.key2id[key] = iset
        start = len(self.seq_parent)
        self.iset_start.append(start)
        self.iset_end.append(start + n_actions)
        self.iset_parent_seq.append(parent_seq)
        self.iset_depth.append(depth)
        self.keys.append(key)
        for _ in range(n_actions):
            self.seq_parent.append(parent_seq)
            self.seq_iset.append(iset)
        return iset

    def finalize(self) -> PlayerTree:
        iset_start = np.asarray(self.iset_start, dtype=np.int64)
        iset_end = np.asarray(self.iset_end, dtype=np.int64)
        iset_depth = np.asarray(self.iset_depth, dtype=np.int64)
        level_seq: List[np.ndarray] = []
        level_iset: List[np.ndarray] = []
        level_offsets: List[np.ndarray] = []
        max_depth = int(iset_depth.max()) if len(iset_depth) else 0
        for d in range(1, max_depth + 1):
            isets = np.where(iset_depth == d)[0]
            isets = isets[np.argsort(iset_start[isets])]
            if len(isets) == 0:
                level_seq.append(np.empty(0, dtype=np.int64))
                level_iset.append(np.empty(0, dtype=np.int64))
                level_offsets.append(np.empty(0, dtype=np.int64))
                continue
            sizes = iset_end[isets] - iset_start[isets]
            offsets = np.concatenate(([0], np.cumsum(sizes)[:-1]))
            seqs = np.concatenate(
                [np.arange(iset_start[i], iset_end[i]) for i in isets]
            )
            level_seq.append(seqs)
            level_iset.append(isets)
            level_offsets.append(offsets.astype(np.int64))
        return PlayerTree(
            n_seq=len(self.seq_parent),
            seq_parent=np.asarray(self.seq_parent, dtype=np.int64),
            seq_iset=np.asarray(self.seq_iset, dtype=np.int64),
            iset_start=iset_start,
            iset_end=iset_end,
            iset_parent_seq=np.asarray(self.iset_parent_seq, dtype=np.int64),
            iset_depth=iset_depth,
            level_seq=level_seq,
            level_iset=level_iset,
            level_offsets=level_offsets,
            iset_keys=list(self.keys),
        )


@dataclass
class EFG:
    """Flattened extensive-form game."""

    n_players: int
    players: List[PlayerTree]
    term_chance: np.ndarray  # [Z] chance reach of each terminal
    term_util: np.ndarray    # [n_players, Z]
    term_seq: np.ndarray     # [n_players, Z] player-i sequence at terminal z

    @property
    def n_terminals(self) -> int:
        return len(self.term_chance)

    def uniform_sigma(self, i: int) -> np.ndarray:
        pt = self.players[i]
        sigma = np.ones(pt.n_seq)
        if pt.n_iset:
            counts = (pt.iset_end - pt.iset_start).astype(float)
            sigma[1:] = 1.0 / counts[pt.seq_iset[1:]]
        return sigma

    def realization(self, i: int, sigma: np.ndarray) -> np.ndarray:
        """Realization plan x[s] = product of sigma along the sequence."""
        pt = self.players[i]
        x = np.ones(pt.n_seq)
        for seqs in pt.level_seq:
            x[seqs] = sigma[seqs] * x[pt.seq_parent[seqs]]
        return x

    def backward_sum(self, i: int, sigma: np.ndarray,
                     term_w: np.ndarray) -> np.ndarray:
        """G[s] = sum over terminals below s of term_w, sigma-weighted below s.

        This is the counterfactual-value primitive: with
        term_w = chance * opp_reach * u_i it returns v(I, a) at each sequence.
        """
        pt = self.players[i]
        g = np.zeros(pt.n_seq)
        np.add.at(g, self.term_seq[i], term_w)
        for d in range(len(pt.level_seq) - 1, -1, -1):
            seqs = pt.level_seq[d]
            if len(seqs) == 0:
                continue
            np.add.at(g, pt.seq_parent[seqs], sigma[seqs] * g[seqs])
        return g

    def best_response_value(self, i: int, term_w: np.ndarray) -> float:
        """Value of the exact best response given term_w = chance*opp_reach*u_i."""
        pt = self.players[i]
        v = np.zeros(pt.n_seq)
        np.add.at(v, self.term_seq[i], term_w)
        for d in range(len(pt.level_seq) - 1, -1, -1):
            seqs = pt.level_seq[d]
            if len(seqs) == 0:
                continue
            best = np.maximum.reduceat(v[seqs], pt.level_offsets[d])
            np.add.at(v, pt.iset_parent_seq[pt.level_iset[d]], best)
        return float(v[0])

    def opp_reach(self, i: int, xs: List[np.ndarray]) -> np.ndarray:
        """chance(z) * prod_{j != i} x_j(z) at every terminal."""
        r = self.term_chance.copy()
        for j in range(self.n_players):
            if j != i:
                r *= xs[j][self.term_seq[j]]
        return r

    def joint_reach(self, xs: List[np.ndarray]) -> np.ndarray:
        """chance(z) * prod_j x_j(z) at every terminal."""
        r = self.term_chance.copy()
        for j in range(self.n_players):
            r *= xs[j][self.term_seq[j]]
        return r

    def local_regret(self, i: int, sigma: np.ndarray,
                     v: np.ndarray) -> np.ndarray:
        """Instantaneous counterfactual regret ir(s) = v(s) - v(I(s))."""
        pt = self.players[i]
        ir = v.copy()
        ir[0] = 0.0
        if pt.n_iset:
            iset_v = np.add.reduceat(sigma * v, pt.iset_start)
            ir[1:] -= iset_v[pt.seq_iset[1:]]
        return ir

    def normalize(self, i: int, vals: np.ndarray) -> np.ndarray:
        """Per-infoset normalization of nonnegative vals; uniform where all zero."""
        pt = self.players[i]
        out = np.zeros(pt.n_seq)
        out[0] = 1.0
        if pt.n_iset == 0:
            return out
        sums = np.add.reduceat(vals, pt.iset_start)
        counts = (pt.iset_end - pt.iset_start).astype(float)
        seq_i = pt.seq_iset[1:]
        pos = sums > 0
        denom = np.where(pos, sums, counts)[seq_i]
        numer = np.where(pos[seq_i], vals[1:], 1.0)
        out[1:] = numer / denom
        return out


def build_efg(game: Any) -> EFG:
    """Flatten a game definition into an EFG via an explicit-stack tree walk.

    The game object must provide ``n_players`` and ``initial_state()``; states
    must provide is_terminal/is_chance/player/infoset_key/legal_actions/
    child/chance_outcomes/utility (see src/games/).
    """
    n = game.n_players
    builders = [_PlayerBuilder() for _ in range(n)]
    term_c: List[float] = []
    term_u: List[List[float]] = []
    term_s: List[tuple] = []
    stack = [(game.initial_state(), 1.0, (0,) * n, (0,) * n)]
    while stack:
        state, chance, seqs, depths = stack.pop()
        if state.is_terminal():
            term_c.append(chance)
            term_u.append(state.utility())
            term_s.append(seqs)
            continue
        if state.is_chance():
            for child, prob in state.chance_outcomes():
                stack.append((child, chance * prob, seqs, depths))
            continue
        i = state.player()
        actions = tuple(state.legal_actions())
        iset = builders[i].infoset(
            state.infoset_key(), seqs[i], depths[i] + 1, actions
        )
        s0 = builders[i].iset_start[iset]
        new_depths = depths[:i] + (depths[i] + 1,) + depths[i + 1:]
        for k, action in enumerate(actions):
            new_seqs = seqs[:i] + (s0 + k,) + seqs[i + 1:]
            stack.append((state.child(action), chance, new_seqs, new_depths))
    efg = EFG(
        n_players=n,
        players=[b.finalize() for b in builders],
        term_chance=np.asarray(term_c),
        term_util=np.asarray(term_u, dtype=float).T,
        term_seq=np.asarray(term_s, dtype=np.int64).T,
    )
    logger.info(
        "built EFG: %d terminals, infosets per player %s",
        efg.n_terminals, [p.n_iset for p in efg.players],
    )
    return efg
