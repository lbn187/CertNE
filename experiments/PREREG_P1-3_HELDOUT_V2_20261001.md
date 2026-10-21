# Preregistration v2 (frozen): selector comparison after the pool extension (seeds 64--127)

English rendering of the frozen protocol document. The original was written
and frozen on 2026-10-01, before any held-out run of the extension was
launched or any such result existed. Machine-specific paths and account
names have been removed; no definition, threshold, band, endpoint or
prediction has been changed.

**Status: frozen (2026-10-01).** The document was filled in and frozen by
the execution session on the user's instruction of the day ("freeze
preregistration v2 after reading the report, then run the held-out stage per
Section 6"); every input came from the full seeds 0--63 report of 2026-10-01
(`ARM_POOL_V2_RESULTS_20261001.md`) and its pooled summary
`pool_plus_all9.json`. At the time of filling, the held-out instances (seeds
64--127) had not been run. After freezing, this document is not modified;
any held-out run is read against this file only. v1
(`PREREG_heldout.md`, the rendering of the 2026-09-09 protocol) stays frozen
and untouched; this document covers only the **pool extension**.

## 1. Purpose

The 7-arm pool of the paper has already been evaluated on seeds 64--127
under v1 (band `[0.74, 0.91]`, matching the pool's in-sample 0.84). After
the extension, the same held-out instances must answer:

- (a) whether the extended-pool C^1 selection solved fraction `f_C1^{v2}`
  falls inside the band **recomputed for the new pool**;
- (b) whether the extension **increment** `f_C1^{v2} - f_C1^{v1}` has the
  same sign on held-out as on in-sample (same 64 instances, paired
  instance-wise, McNemar exact test);
- (c) whether C^1 selection still agrees instance-wise with same-grid
  exact-gap selection (`S_EXACT`) (v1 reported 54 = 54).

## 2. Data-generating process (inherited from v1, unchanged)

- **Game**: biased two-player Leduc,
  `beta = float(np.random.default_rng(seed).uniform(0.0, 0.5))`,
  `seed in {64, ..., 127}`; `T = 2**18`, `eval_every = 2048` (128
  checkpoints); `strict_regret_logging=False` (same convention as
  `m1_heldout_seq`); Python 3.12.9 + NumPy 2.1.2; one CPU node with BLAS/OMP
  thread counts = 1.
- **New-arm logs**: `results/m1_pool_v2_heldout/<arm>_seed<N>.json`,
  produced by `experiments/run_arm_pool_v2.py --stage heldout --prereg
  <this file>`. The held-out logs of the 7 paper arms are reused from
  `results/m1_heldout_seq/` (not rerun).

## 3. Extended arm table (frozen)

**Frozen decision: all nine selected arms enter the pool, with no second
screening round.** Each arm was chosen family-wise by the selection rule
frozen at the pilot stage (`run_arm_pool_v2.FULL_ARMS`, fixed before the
full-stage results); taking all nine gives a zero-degree-of-freedom arm
table.

| arm (key in `run_arm_pool_v2.ARMS`) | family | in-sample oracle@1e-5 (seeds 0--63, counts) | reason for inclusion (single-arm union@1e-5; 7-arm pool = 54/64 union, 48/64 cert) |
|---|---|---|---|
| `dcfr_alt_w2` | dcfr | 0/64 | discounting family (reviewers will ask); single-arm union 54/64, cert 48/64 |
| `dcfr+_alt_a1.5_w4` | dcfr+ | 0/64 | discounting family; single-arm union 54/64, cert 48/64 |
| `pdcfr+_alt_a1.5_w5` | pdcfr+ | 42/64 | discounting family; single-arm cert 49/64 |
| `hs-pcfr+_alt_g30` | hs-pcfr+ | 46/64 | single-arm cert 49/64 |
| `hs-dcfr_alt_g15` | hs-dcfr | 0/64 | HS-DCFR family representative; single-arm union 54/64, cert 48/64 |
| `sprm+_alt_eta10000_w2` | sprm+ | 47/64 | single-arm cert 51/64 |
| `stprm+_alt_eta1000_w2` | stprm+ | 51/64 | single-arm cert 53/64 |
| `exrm+_sim_eta10000_w2` | exrm+ | 17/64 | single-arm union 56/64 (paper pool 54/64, +2) |
| `irprm+_alt_w2` | irprm+ | 12/64 | single-arm union 55/64 (paper pool 54/64, +1) |

- The draft's "suggested" column filter `oracle@1e-5 >= 0.60` is **not
  adopted**: that gate screens single arms, while the pool's gain is decided
  by the union -- `exrm+` and `irprm+` are the only two candidates that add
  to the union (54/64 -> 56/64 / 55/64), and a >= 0.60 filter would drop
  both, contradicting the pool-extension conclusion of report Section 2.
- With all nine arms in (7 paper arms union 9 new arms = 16 arms), the
  in-sample triple (`pool_plus_all9.json`, @1e-5, 64 seeds) is: oracle union
  **0.890625**, C1 selection **0.890625**, certification **0.890625**
  (recorded as 0.89 / 0.89 / 0.89 in the report).
- Once listed, no arm may be added or removed after held-out results appear.

The new pool is `P_v2 = the 7 paper arms union the table above`.

## 4. Frozen selector definitions (verbatim from v1 Section 3, extended to the new pool)

For arm `a in P_v2`, instance `s`, checkpoint `t in {0, ..., 127}`:
`gap[a,s,t]` exact NashGap; `seq[a,s,t]` = `certificate_seq` (C^1);
`seq_l1` = `certificate_seq_l1` (C^0);
`regret[a,s,t] = terms[a,s,t,:,0].max()`.

Cross-arm selection: `(a*, t*) = argmin_{a,t} score[a,s,t]`, reporting
`gap[a*,s,t*]`. `S_C1` (score = seq), `S_REG` (score = regret),
`S_EXACT = min_{a,t} gap`. Certification score
`cert_C1(s) = min_{a,t} seq[a,s,t]`. **The threshold is fixed at 1e-5**.

## 5. Primary endpoint and band (frozen)

- **Primary endpoint**: `f_C1^{v2} = (1/64) sum_s [gap at S_C1 <= 1e-5]`
  (new pool).
- **Band**: the full-stage in-sample point estimate of the new pool is
  `p_v2 = 0.890625` (= 57/64, `n = 64`; recorded as 0.89).
  `1.96*sqrt(p(1-p)/64) = 0.07646673698566177`, so
  `p_v2 +/- 1.96*sqrt(p_v2(1-p_v2)/64) = [0.8141583, 0.9670917]`, rounded to
  0.01 gives **[0.81, 0.97]**. **This band was written before any held-out
  run of a new arm was launched (the freeze point of this file).**
- **Reading rule**: inside the band -> the extension transfers; below the
  lower bound -> report the measured value, do not change the selector,
  threshold or arm table; above the upper bound -> report it as a favourable
  draw and report `S_EXACT` and `S_REG` alongside.
- **Paper convention**: the in-sample and held-out numbers are stated side
  by side; the held-out value must not silently replace the in-sample value.

## 6. Secondary, directional predictions (frozen)

| Id | Prediction | Basis |
|---|---|---|
| S1 | `S_EXACT >= S_C1` | definition (pointwise minimum on the same grid) |
| S2 | `S_C1 >= S_REG` | v1 direction measured on held-out |
| S3 | `f_cert <= f_C1` | definition |
| S4 | the new pool's `S_C1` and `S_EXACT` give the same solved-or-not decision on >= 60/64 instances | v1: 54 = 54 |
| S5 | paired increment: instances the new pool solves and the 7-arm pool does not (`n+`) versus the reverse (`n-`), with `n+ >= n-` | in-sample increment direction |
| S6 | in the new pool's selected-arm distribution, the count of selections landing on a new arm is >= **55** | in-sample `selected_arms`: new arms 59/64; threshold = 59 - 1.96*sqrt(64*(59/64)*(5/64)) = 54.79, rounded |

## 7. Analysis (no new statistic)

1. `experiments/analyze_pool_v2.py --dir results/m1_pool_v2_heldout
   --paper-dir results/m1_heldout_seq --pool-extra <arm table>`
   (`--paper-dir` reads the 7 paper arms from the held-out logs). Read
   `pool_plus_extra`'s `c1.selection[-1]`, `c1.certification[-1]`,
   `oracle_union[-1]`, `c1.selected_arms`.
2. Paired increment: compare the `S_C1` solved indicator of the v1 pool
   and the v2 pool instance-wise, McNemar exact two-sided p-value; report
   only `n+`, `n-` and p, with no one-sided reading.
3. Code bug -> record, fix, **rerun the full grid**; no per-instance
   patches.

## 8. Run and artifacts

- Entry: `run_pool_v2_cluster.sh 60 --stage heldout --arms <Section 3 arm list>
  --prereg <this file>`; 60 workers; artifacts
  `results/m1_pool_v2_heldout/<arm>_seed<N>.json` (64 per arm).
- Budget: about 2--7 core-h per arm (experiment plan Section 0), total linear in
  the number of arms.
- Idempotent: existing JSON files are skipped. Isolation: directory kept
  separate from the in-sample one.

## 9. Known risks

1. `sprm+`/`exrm+` trajectories are sensitive to last-bit floating point;
   run them only in the same cluster environment; numerics across
   environments are not a conclusion.
2. S1/S3 are identities; they are consistency checks only.
3. If a new arm's held-out oracle falls well below its in-sample value
   (> 0.15), report it as measured; do not adjust the arm table.
