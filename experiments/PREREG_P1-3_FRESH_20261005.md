# Preregistration v3 (frozen): sixteen-arm pool validation on a fresh instance set (seeds 128--191)

English rendering of the frozen protocol document. The original was written
and frozen on 2026-10-05, before any fresh-instance run was launched or any
such result existed. Machine-specific paths and account names have been
removed; no definition, threshold, band, endpoint or prediction has been
changed.

**Status: frozen (2026-10-05).** Every input came from the seeds 0--63
in-sample report and the seeds 64--127 v2 held-out results; at the time of
writing, **seeds 128--191 had not been run** (the two output directories
`results/m1_fresh_seq/` and `results/m1_pool_v2_fresh/` did not exist). After
freezing, this document is not modified; any fresh run is read against this
file only. v1 (`PREREG_P1-3_HELDOUT_20260909.md`) and v2
(`PREREG_P1-3_HELDOUT_V2_20261001.md`) stay frozen and untouched; this
document covers only the **fresh instances**.

## 1. Purpose

Reviewers pointed out that the v2 held-out set (seeds 64--127) had already
been used in Stage 1 (seven arms), so Stage 2's reuse of it is **not fresh
data**, and the v2 59/64 cannot count as independent replication evidence.
This preregistration reruns the **full sixteen-arm protocol** on an instance
set used by no earlier stage (seeds 128--191), with the protocol, arm table,
selector and threshold kept verbatim and only the instances changed,
answering:

- (a) whether the primary endpoint `f_C1^{v3}` falls inside the band written
  for the new pool (whether the pool-extension conclusion reproduces on a
  third instance set);
- (b) how large the point-estimate difference between fresh and held-out
  (seeds 64--127, measured 0.921875) is (reading only).

## 2. Data-generating process (identical to the v2 held-out protocol, verbatim)

- **Game**: biased two-player Leduc,
  `beta = float(np.random.default_rng(seed).uniform(0.0, 0.5))`,
  `seed in {128, ..., 191}`; `T = 2**18`, `eval_every = 2048` (128
  checkpoints); `strict_regret_logging=False`; Python 3.12.9 + NumPy 2.1.2;
  one CPU node with BLAS/OMP thread counts = 1.
- **Logs**:
  - 7 paper arms -> `results/m1_fresh_seq/<arm>_seed<N>.json`, produced by
    `experiments/run_m1_heldout.py --seed-start 128 --n-seeds 64 --out-dir
    results/m1_fresh_seq` (**the same runner and the same code as
    `m1_heldout_seq`**, with only the instance range and the output
    directory changed).
  - 9 new arms -> `results/m1_pool_v2_fresh/<arm>_seed<N>.json`, produced by
    `experiments/run_arm_pool_v2.py --stage fresh --prereg <this file>`.

## 3. Arm table (frozen, verbatim from v2)

**The 7 paper arms** (`run_m1_grid.build_arms(house=False, portfolio=True)`):

`rm+_alt_w1`, `rm+_alt_w2`, `pcfr+_alt_w1`, `pcfr+_alt_w2`,
`dprm+_alt_w1_lam0.3`, `dprm+_alt_w2_lam0.1`, `dprm+_alt_w2_lam0.3`

**The 9 extended-pool arms** (`run_arm_pool_v2.FULL_ARMS`, frozen at v2):

| arm | family |
|---|---|
| `dcfr_alt_w2` | dcfr |
| `dcfr+_alt_a1.5_w4` | dcfr+ |
| `pdcfr+_alt_a1.5_w5` | pdcfr+ |
| `hs-pcfr+_alt_g30` | hs-pcfr+ |
| `hs-dcfr_alt_g15` | hs-dcfr |
| `sprm+_alt_eta10000_w2` | sprm+ |
| `stprm+_alt_eta1000_w2` | stprm+ |
| `exrm+_sim_eta10000_w2` | exrm+ |
| `irprm+_alt_w2` | irprm+ |

The new pool `P_v3 = the 7 arms union the 9 arms` (16 arms) is exactly v2's.
**Only the instances change; nothing else.**

## 4. Selector and threshold (cast from v2 Section 4, unchanged)

For arm `a in P_v3`, instance `s`, checkpoint `t in {0, ..., 127}`:
`gap[a,s,t]` exact NashGap; `seq[a,s,t]` = `certificate_seq` (C^1);
`seq_l1` = `certificate_seq_l1` (C^0);
`regret[a,s,t] = terms[a,s,t,:,0].max()`.

Cross-arm selection: `(a*, t*) = argmin_{a,t} score[a,s,t]`, reporting
`gap[a*,s,t*]`. `S_C1` (score = seq), `S_REG` (score = regret),
`S_EXACT = min_{a,t} gap`. Certification score
`cert_C1(s) = min_{a,t} seq[a,s,t]`. **The threshold is fixed at 1e-5**.

## 5. Primary endpoint and band (frozen)

- **Primary endpoint**: `f_C1^{v3} = (1/64) sum_s [gap at S_C1 <= 1e-5]`
  (16-arm pool).
- **Band**: **[0.81, 0.97]**, the same definition as v2 Section 5, from the
  Stage 1 in-sample point estimate `p_v2 = 0.890625` (57/64):
  `1.96*sqrt(p_v2(1-p_v2)/64) = 0.0764667`,
  `[0.8141583, 0.9670917]` rounded to 0.01. **This band was written before
  any fresh run was launched (the freeze point of this file).** The band is
  inherited from v2 rather than taken from the held-out measurement so that
  the two stages share one primary criterion definition, avoiding the
  freedom of setting a band on the second batch and validating on a third.
- **Reading rule**: inside the band -> the pool-extension result reproduces
  on a third, independent instance set; below the lower bound -> report the
  measured value, do not change the selector, threshold or arm table; above
  the upper bound -> report it as a favourable draw and report `S_EXACT`
  and `S_REG` alongside.
- **Paper convention**: the numbers of the three instance sets (in-sample
  0--63, held-out 64--127, fresh 128--191) are stated side by side; no
  batch may silently replace another.

## 6. Secondary, directional predictions (frozen)

| Id | Prediction | Basis |
|---|---|---|
| S1 | `S_EXACT >= S_C1` | definition (pointwise minimum on the same grid) |
| S2 | `S_C1 >= S_REG` | direction measured on both v1 and v2 |
| S3 | `f_cert <= f_C1` | definition |
| S4 | the new pool's `S_C1` and `S_EXACT` give the same solved-or-not decision on >= 60/64 instances | v1: 54 = 54; v2 held-out: instance-wise agreement |
| S5 | paired increment: instances the new pool solves and the 7-arm pool does not (`n+`) versus the reverse (`n-`), with `n+ >= n-` | direction measured on both batches |
| S6 | in the new pool's selected-arm distribution, the count of selections landing on a new arm is >= **55** | v2 threshold, unchanged |
| S7 | (new; reading only, no criterion) the difference between the fresh `f_C1` and the held-out `0.921875`, reported as a point estimate and a Wilson/normal 95% interval for the difference of two proportions | cross-instance-set reproducibility |

## 7. Analysis (no new statistic)

1. `experiments/analyze_pool_v2.py --dir results/m1_pool_v2_fresh
   --paper-dir results/m1_fresh_seq --pool-extra <the 9 arms of Section 3>`.
   Read `pool_plus_extra`'s `c1.selection[-1]`, `c1.certification[-1]`,
   `oracle_union[-1]`, `c1.selected_arms`.
2. Paired increment and cross-set comparison: compare the `S_C1` solved
   indicator of the 7-arm pool and the 16-arm pool instance-wise, McNemar
   exact two-sided; for the fresh-versus-held-out proportion difference
   report only the point estimate and the interval, with no one-sided
   reading.
3. Code bug -> record, fix, **rerun the full grid**; no per-instance
   patches.

## 8. Run and artifacts

- Two cluster jobs, one CPU node each, 60 workers each:
  - `run_m1_fresh_cluster.sh 60 128 64` (7 arms, 448 runs);
  - `run_pool_v2_fresh_cluster.sh 60` (9 arms, 576 runs).
- Budget: about 22 core-h, extrapolated from the held-out measurements
  (7 arms 7.8 core-h + 9 arms 13.6 core-h); about 0.4 h at 60 processes.
- Idempotent: existing JSON files are skipped. Isolation: two new
  directories, no cross-writing with any existing directory.

## 9. Known risks

1. `sprm+`/`exrm+` trajectories are sensitive to last-bit floating point ->
   both jobs run in the same cluster environment; numerical differences
   across environments are not a conclusion.
2. S1/S3 are identities; they are consistency checks only.
3. If an arm's fresh oracle falls well below its value on the two earlier
   batches (> 0.15), report it as measured; do not adjust the arm table.
4. If the three batches' `f_C1` differ beyond the band, report and explain
   as measured; no extra rerun, no replacement instance set.
