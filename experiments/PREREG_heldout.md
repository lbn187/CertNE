# Preregistration: selector comparison on the held-out instance set (seeds 64--127)

English rendering of the frozen protocol document. The original was written
on 2026-09-09, before any held-out run was launched or any held-out result
existed. Machine-specific paths and account names have been removed; no
definition, threshold, band, endpoint or prediction has been changed.

## 1. Purpose

On 64 instances never used for any modelling, tuning or selection decision,
test three things:

- **(a)** whether the main-table `0.84` (C^1 selection solved fraction at
  `1e-5`) transfers to an independent instance sample;
- **(b)** the relation between C^1 selection and "exact-gap selection on the
  same grid" (the paper states the two coincide, 54/64 = 54/64);
- **(c)** the relation between C^1 selection and the regret-only selector
  (the paper states 0.83, one instance fewer).

## 2. Data-generating process (frozen)

- **Game**: biased Leduc (two players); per-instance tie-recovery fraction
  `beta = float(np.random.default_rng(seed).uniform(0.0, 0.5))`,
  `seed in {64, ..., 127}`. None of these seeds appears in any earlier run,
  selection or tuning of the project (checked: zero result files with
  `seed >= 64`; the existing seed range ends at 63).
- **Horizon**: `T = 2**18`, `eval_every = 2048`, i.e. 128 checkpoints per
  instance.
- **Arm pool**: `build_arms((0.1, 0.3), house=False, portfolio=True)`,
  seven arms, all alternating and training-free: `rm+_alt_w1`, `rm+_alt_w2`,
  `pcfr+_alt_w1`, `pcfr+_alt_w2`, `dprm+_alt_w2_lam0.1`,
  `dprm+_alt_w2_lam0.3`, `dprm+_alt_w1_lam0.3`.
- **Recorded fields**: `t`, `nash_gap`, `player_gaps`, `corr_kl`,
  `certificate`, `avg_regret_bound`, `certificate_tight`,
  `certificate_tight_strict`, `avg_regret_bound_strict`, `certificate_seq`
  (C^1), `certificate_seq_l1` (C^0), `certificate_seq_md`,
  `certificate_seq_terms` (per player `[Rhat, Delta, L, ||w||_1, M*D]`).
- **`strict_regret_logging=False`**: the aligned comparison is not tested in
  this round, so the aligned start-of-round certificate is not logged. This
  matches the in-sample `m1_tight_seq` configuration exactly.
- **Engine and interpreter**: the released `code/src`; Python 3.12.9 with
  NumPy 2.1.2, identical to the in-sample runs (the paper records the
  last-bit sensitivity of DPRM+ on Goofspiel).
- **Parallelism**: one CPU node, 32 workers, all BLAS/OMP thread counts = 1.

## 3. Frozen selector definitions (verbatim from `experiments/analyze_m1_seq.py`)

For arm `a`, instance `s`, checkpoint index `t in {0, ..., 127}`:

- `gap[a,s,t]` = recorded exact NashGap;
- `seq[a,s,t]` = `certificate_seq` (C^1);
- `seq_l1[a,s,t]` = `certificate_seq_l1` (C^0);
- `regret[a,s,t]` = `terms[a,s,t,:,0].max()` (i.e. `max_i Rhat_i`, regret
  term only).

Cross-arm (portfolio) selection: take `(a*, t*) = argmin_{a,t} score[a,s,t]`
and report `gap[a*, s, t*]`, the exact gap of the selected iterate.

Three selectors (**frozen**):

| Symbol | Definition | Implementation |
|---|---|---|
| `S_C1` | portfolio selection with `score = seq` | `ablation.portfolio["seq"]` |
| `S_REG` | portfolio selection with `score = regret` | `ablation.portfolio["regret"]` |
| `S_EXACT` | `min_{a,t} gap[a,s,t]` (same-grid exact-gap selection) | `ablation.portfolio["oracle"]`, equal to `portfolio_rows()["oracle_union"]` |

`ablation.portfolio[k]`: for each instance,
`best = min((score(runs[a][s], k).min(), a) for a in pool)` picks the arm,
then `argmin_t score` within that arm, and the pick counts if `gap <= th`;
the `oracle` key is `min_a min_t gap[a,s,t] <= th`.

Certification score (**frozen**): `cert_C1(s) = min_{a,t} seq[a,s,t]`,
i.e. `portfolio_rows()["seq"]["certification"]`.
**The threshold is fixed at `1e-5`** and is not adjusted after the fact.

## 4. Primary endpoint and preregistered band

- **Primary endpoint**: `f_C1 = (1/64) sum_s [ gap at S_C1 <= 1e-5 ]`, the
  C^1-selection solved fraction.
- **Preregistered band `[0.74, 0.91]`**: the rounded binomial normal
  approximation 95% interval around the in-sample point estimate `p = 0.84`,
  `n = 64` (`0.84 +/- 1.96 * sqrt(0.84 * 0.16 / 64) ~ 0.84 +/- 0.09`). The band
  was written **before any held-out result was seen**.
- **Reading rule (frozen)**:
  - `f_C1 in [0.74, 0.91]` -> the main-table `0.84` transfers to an
    independent instance sample;
  - `f_C1 < 0.74` -> transfer fails; report the measured value, **do not
    change the selector or the threshold**;
  - `f_C1 > 0.91` -> report as a favourable draw and report `S_EXACT` and
    `S_REG` alongside.
- Whatever the outcome, the paper labels `0.84` as "in-sample, 64 instances"
  and adds the held-out result in the main text or appendix; the held-out
  value must not silently replace the in-sample value.

## 5. Secondary, directional predictions (frozen)

| Id | Prediction | Basis |
|---|---|---|
| S1 | `S_EXACT >= S_C1` | `S_EXACT` is the pointwise minimum on the same grid, hence an upper bound on the C^1 pick by definition |
| S2 | `S_C1 >= S_REG` | paper: 0.84 vs 0.83 |
| S3 | `f_cert <= f_C1` | the certification score is the selection score's minimum; the gap at the argmin can be below the certified value |
| S4 | `S_C1` and `S_EXACT` give the same solved-or-not decision on >= 60/64 instances | paper: full agreement (54 = 54) |
| S5 | median beta of the failures < median beta over all instances | paper: 0.16 vs 0.28 |

S1/S3 hold by definition; they are reported but not counted as transfer
evidence.

## 6. Analysis

1. The main result is recomputed by the released script, not hand-tabulated:
   `python3 experiments/analyze_m1_seq.py --dir <heldout_dir> --no-regression`,
   reading `ablation.portfolio` keys `seq` / `regret` / `oracle` at `1e-5`.
2. Per-instance detail (S4/S5, beta distribution of failures) is recomputed
   from the same JSON files with the same script's `ablation()`; no new
   statistic is introduced.
3. **No post-hoc tuning**: threshold, arm pool, selector definitions, band and
   endpoint are frozen here. If a code bug is found it is recorded, fixed, and
   the **full grid is rerun**; affected instances are not patched individually.

## 7. Run and artifacts

- Entry point: `experiments/run_m1_heldout.py` (seeds 64--127, `T = 2**18`,
  `eval_every = 2048`, `strict_regret_logging=False`), 32 workers on one node.
- Artifacts: `results/m1_heldout_seq/<arm>_seed<N>.json` (448 files) and
  `summary_seq.json`.
- Budget: same scale as in-sample, about 15--20 core-hours.
- Idempotent: existing JSON files are skipped, so an interrupted run resumes.
- Isolation: output directory `m1_heldout_seq`, separate from the in-sample
  `m1_tight_seq` (the script's default output directory was set before
  submission so that in-sample results cannot be overwritten).

## 8. Known risks

1. **Environment**: same interpreter and NumPy build as in-sample; the entry
   script was smoke-tested (`--smoke`, 14 runs) before submission.
2. **S1/S3 are tautologies**: reported as consistency checks only.
3. **beta sampling**: drawn by `default_rng(seed)` with the same distribution
   and convention as seeds 0--63, on disjoint seeds.
4. **Failure mode**: if `f_C1` falls well below the band, the paper reports it
   and restricts the conclusion to the in-sample/held-out discrepancy.

**Freeze**: this document was written to disk before submission. Any later
change to a selector, threshold, band or endpoint requires a separate
document with a stated reason; the frozen text is not edited in place.
