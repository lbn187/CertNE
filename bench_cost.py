#!/usr/bin/env python3
"""Per-checkpoint cost micro-benchmark behind Table S9 (tab:budget).

Measures, on one core and biased Leduc (beta=0.25): one training iteration,
an exact NashGap evaluation, the C^0 and C^1 certificate terms, and the
extra cost of the aligned start-of-round pass. Run from the code root:

    python3 bench_cost.py
"""
import sys, time
import numpy as np
sys.path.insert(0, '.')
from src.games.biased import make_biased_leduc
from src.efg import build_efg
from src import metrics
from src.cfr import CFRConfig, run_cfr

efg = build_efg(make_biased_leduc(0.25))
n = efg.n_players
print('players', n, 'terminals', efg.n_terminals,
      'seqs', [pt.n_seq for pt in efg.players], flush=True)

T = 2000
cfg = CFRConfig(alg='pcfr+', t_max=T, weight_exp=2.0, update='alternating',
                eval_every=10**9, checkpoints=(T,))
t0 = time.perf_counter(); res = run_cfr(efg, cfg)
it_ms = (time.perf_counter() - t0) / T * 1e3
cfg2 = CFRConfig(alg='pcfr+', t_max=T, weight_exp=2.0, update='alternating',
                 eval_every=10**9, checkpoints=(T,), strict_regret_logging=True)
t0 = time.perf_counter(); run_cfr(efg, cfg2)
it_strict_ms = (time.perf_counter() - t0) / T * 1e3

xs_bar = res['xs_bar']; rc = res['regret_cum']
def bench(f, k=50):
    f(); t0 = time.perf_counter()
    for _ in range(k):
        f()
    return (time.perf_counter() - t0) / k * 1e3
w = np.random.randn(efg.n_terminals)
d = [np.random.rand(efg.n_terminals) for _ in range(n)]
c = [np.random.rand(efg.n_terminals) for _ in range(n)]
reg_bounds = [metrics.cfr_regret_bound(efg, i, rc[i]) / T for i in range(n)]
# The engine's checkpoint cost is exactly certificate_seq_terms plus the
# regret bounds; time those two calls.
def c0():
    metrics.certificate_seq_terms(efg, xs_bar, d, c, reg_bounds, zero_br=True)
def c1():
    metrics.certificate_seq_terms(efg, xs_bar, d, c, reg_bounds, zero_br=False)

print('per-iteration, alternating PCFR+ (C^0/C^1 accumulators on): %.3f ms' % it_ms)
print('per-iteration, with aligned start-of-round regret pass:     %.3f ms' % it_strict_ms)
print('exact NashGap (n best-response passes):                     %.3f ms'
      % bench(lambda: metrics.nash_gap(efg, xs_bar)))
print('one signed backward pass (C^1 comparator L):                %.3f ms'
      % bench(lambda: efg.best_response_value(1, w)))
print('C^0 checkpoint (certificate_seq_terms, zero_br=True):       %.3f ms'
      % bench(c0))
print('C^1 checkpoint (certificate_seq_terms, exact L):            %.3f ms'
      % bench(c1))
