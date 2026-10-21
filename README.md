# CertNE

Code for the paper **When Are We Near Nash? Certifying Solutions in
General-Sum Games** by Boning Li and Longbo Huang.

CertNE builds an online NashGap certificate from a solver's own regret and
terminal-distribution accumulators, then selects the iterate with the
smallest certificate across a portfolio of training-free regret-matching
solvers. This repository regenerates the paper's experiments: the biased
Leduc in-sample, held-out and fresh grids, the pool extension, three-player
Leduc, Sheriff and Goofspiel, and the cost benchmarks.

The core modules under `src/` are readable Python: they are the sources the
reported runs' compiled builds were generated from. The entry scripts under
`experiments/` are Python too, so the commands below can be read and rerun
as written. Unit tests of `src/` ship in `tests/`. No main-run logs are
bundled; the certificate-gated control experiment ships as audit logs only
(see `MANIFEST.md` §6). The paper's figures are regenerated from run logs
with the scripts in `figures/`.

The portfolio has 16 training-free solvers: seven original and nine added in
a second pre-registered stage, one per family (DCFR, DCFR+, PDCFR+, HS-PCFR+,
HS-DCFR, Smooth/Stable/ExRM+ regret matching+, IR-PRM+). The 7-solver and
16-solver numbers come from the same logs; the paper's Table 1 keeps both
portfolios, and the nine added solvers use the parameter values printed in
the appendix.

`MANIFEST.md` maps each table and figure label to its command and states
what ships with the package.

## Quick start

Requirements: Python 3.10 or later and NumPy. The reported runs used Python
3.12.9 and NumPy 2.1.2 (`requirements.txt`). No GPU is needed; the run
scripts use a multiprocessing pool.

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests -q
python3 experiments/run_m1_seq.py --smoke --workers 2
python3 experiments/analyze_m1_seq.py --dir results/m1_tight_seq_smoke --no-regression
```

The smoke run exercises the seven original solvers on two instances at a
reduced budget and writes to `results/m1_tight_seq_smoke`. Every run script
skips a run whose log already exists, so an interrupted grid resumes at run
granularity. The full grids are described under **Full reproduction** below.

## Figures

The figure scripts are in `figures/` and write next to themselves. They read
the run logs in `results/`; override the input directories with `TBD_M1_DIR`,
`TBD_POOL_V2_DIR`, or `TBD_LEDUC3_JSON`. `gen_fig_protocol_solvers.py` is a
schematic and needs no data, `gen_fig_tightness_v3.py` needs the seven- and
sixteen-solver in-sample logs, and `gen_fig_coalition.py` needs the
three-player Leduc log. Figure dependencies are in `requirements-dev.txt`.

## Data and logs

Run logs are large and are not bundled. Every run script skips a run whose
log already exists, so an interrupted grid resumes at run granularity; there
is no checkpoint-and-resume inside a single solver run. The analysis scripts
read the logs and print the paper's numbers. `MANIFEST.md` §2 lists which
directories each command regenerates, and §6 lists the results that this
package does not reproduce.

## Requirements

Python 3.10 or later with NumPy (`requirements.txt` pins 2.1.2, which itself
needs Python >= 3.10; the reported runs used Python 3.12.9 for the paper arms
and Python 3.10/3.12 for the extension). No GPU is needed; the run scripts
use a multiprocessing pool. A full reproduction is about 166 core-hours of
summed per-run wall-clock over all grids of this package (two-player
in-sample, held-out, fresh and external grids, the selection pilot, and the
three-player runs); the three-player runs are the slowest (about two to three
core-hours per arm).

The reported runs used compiled builds generated from the shipped `src/`
sources. The compiled and the source tree were verified to produce
byte-identical result trees, and rerunning the same compiled binaries on one
machine under CPython 3.10.21 (NumPy 2.2.6), 3.12.3 (NumPy 1.26.4) and 3.12.9
(NumPy 2.1.2) produces byte-identical logs for the seven biased-Leduc paper
arms at seed 0 (MANIFEST.md §3). Running from source needs no build step, and
the end-to-end wait is dominated by the solver.

## Layout

```
CertNE/
├── README.md
├── MANIFEST.md                label -> command map; bundled vs regenerated
├── requirements.txt           environment pin
├── requirements-dev.txt       tests and figure scripts
├── src/                       core, readable Python:
│   ├── efg.py                 sequence-form EFG engine
│   ├── cfr.py                 CFR/PCFR+/DPRM+ loop and logging
│   ├── decorr.py              DPRM+ decorrelation direction
│   ├── metrics.py             certificates (C^1, C^0, M*D), NashGap, KL
│   ├── variants.py            the added regret-minimizer variants
│   └── games/                 biased Leduc, three-player Leduc, Kuhn,
│                              Sheriff, Goofspiel, matrix games
├── tests/                     unit tests of src/ (`python3 -m pytest tests`)
│   └── ref_src_20261001/      frozen source snapshot for the legacy
│                              bit-identical checks
├── experiments/
│   ├── run_m1_grid.py         arm definitions (build_arms) and BASE_DIR
│   ├── run_m1_seq.py          7-arm biased-Leduc grid -> results/m1_tight_seq
│   ├── run_m1_heldout.py      same grid on seeds 64--127 -> results/m1_heldout_seq;
│   │                          --seed-start 128 writes the fresh 7-arm set
│   ├── PREREG_heldout.md      frozen protocol of the held-out experiment (v1)
│   ├── PREREG_P1-3_HELDOUT_V2_20261001.md
│   │                          frozen protocol of the pool extension (v2)
│   ├── PREREG_P1-3_FRESH_20261005.md
│   │                          frozen protocol of the fresh set (v3)
│   ├── run_extra_arms.py      three control arms -> results/m1_extra
│   ├── run_arm_pool_v2.py     the nine added arms (pilot/full/held-out/
│   │                          fresh stages)
│   │                          -> results/m1_pool_v2[_heldout/_fresh]
│   ├── run_external.py        Sheriff / Goofspiel, 7 arms -> results/ext_<game>_seq
│   ├── run_external_pool_v2.py  the nine added arms on both games
│   │                          -> results/ext_<game>_pool_v2
│   ├── run_leduc3_seq.py      three-player Leduc, six arms -> results/leduc3_seq
│   ├── run_leduc3_pool_v2.py  three-player Leduc, six added arms
│   │                          -> results/leduc3_pool_v2
│   ├── analyze_m1_seq.py      Table 1 output rows, pool/per-arm tables,
│   │                          soundness, tightness (also for the external games)
│   ├── analyze_extra_arms.py  control-arm rows of Table 1
│   ├── analyze_pool_v2.py     pool-extension diagnostics (full / held-out)
│   ├── analyze_pool16_paper.py  every two-player number of the paper
│   │                          recomputed from the 16-arm logs,
│   │                          -> results/pool16_paper (with a 7-arm self-check)
│   ├── analyze_fresh_v3.py    stage-3 readings of the v3 pre-registration
│   │                          -> results/pool16_paper
│   ├── analyze_selection_loss.py  selected-gap vs trajectory minimum over
│   │                          the three instance sets (tab:selection-loss)
│   ├── analyze_pool16_leduc3.py  three-player and external numbers
│   ├── analyze_leduc3_seq.py  three-player table (6 paper arms)
│   ├── pool16_common.py       shared constants of the 16-arm analysis
│   ├── confirm_statistics.py  Wilson/McNemar helpers used by the analysis
│   └── side_experiment/       audit logs of the certificate-gated control
│                              experiment quoted in Future Work (logs only;
│                              the solver that produced them is not bundled)
├── bench_cost.py              per-checkpoint wall-clock (the compute-budget
│                              table, `tab:budget`)
├── bench_cost_e2e.py          end-to-end wall-clock of the checkpoint policies
│                              (uninstrumented / certificate state only / C^0 /
│                              C^1 / exact gap / bare + exact gap)
└── figures/
    ├── gen_fig_protocol_solvers.py  protocol schematic (no data)
    ├── gen_fig_tightness_v3.py      tightness, curves and hard-instance panels
    └── gen_fig_coalition.py         three-player coalition certificate
```

Run every command from the code root (`cd code`). Run scripts write into
`results/<tag>/` and skip any run whose log already exists, so interrupted
runs resume. Analysis scripts only read the logs.

> Resuming with a different logging mode: the strictness flag (`--no-strict`,
> aligned logging) is part of a run's log identity. To rerun the same tag with
> a different strictness, pass a fresh `--out-dir` (or delete the old log);
> reusing a log silently skips the run. RM$^+$/PRM$^+$ arms reproduce
> bit-for-bit across NumPy builds; the DPRM$^+$ Goofspiel trajectories are
> sensitive to last-bit floating-point differences, so fix the NumPy version
> used for the Goofspiel rows (Goofspiel oracle-fraction differences up to
> $\pm 0.06$ have been observed across builds).

## Wall-clock behind the compute-budget table (tab:budget)

```
python3 bench_cost.py
```

prints, on one core and biased Leduc (beta=0.25): one training iteration
(~0.37 ms, ~0.51 ms with the aligned pass), one exact NashGap evaluation
(~0.15 ms), and the C^0 (~0.12 ms) and C^1 (~0.19 ms) checkpoint terms. These
are isolated terms timed in a tight loop on warm inputs; inside a run they are
1.2--1.7x larger.

```
python3 bench_cost_e2e.py --grids 2048,8
```

runs the same configuration with `record_mode` `none` / `c0` / `c1` / `gap`
plus two bare-solver arms (`certificate_state=False`: no raw signed regret
sums, no evaluation-sequence terminal distributions, no joint terminal
distribution; the solver is otherwise unchanged and its final average is
bit-identical): `bare` with the in-run selection rule, and `baregap`, which
selects the checkpoint with the smallest exact gap seen so far (the
trajectory-minimum rule, run online). It reports wall-clock, timed checkpoint
work and peak RSS per arm; `--repeats 3` runs each arm three times in shuffled
order and reports mean and min--max.
At the protocol grid (`eval_every=2048`) the four certificate-state policies
differ by less than run-to-run variation (116--119 s over three repeats) and
the checkpoint work is under 0.05% of the run, while the uninstrumented arm
takes 92 s in the same batch: maintaining the certificate accumulators costs
25.4 s, 21.7% of the instrumented run (27.6% on top of the bare solver) in
this implementation, which is where the certificate's price sits. A later
batch, `--arms bare,c0,baregap`, gives 81.9 s [80.9, 82.9], 94.4 s
[93.9, 95.0] and 84.4 s [82.0, 87.9] for the three arms, with an identical
final average across them; the timed checkpoint work of `baregap` is 0.02 s.
At
`eval_every=8` the timed checkpoint work of the certificate is 6--9% of the
run at up to 62 MB of peak RSS. The "share" and "ms/checkpoint" columns are
the timed checkpoint block, not differences between runs.

## What is logged

`src/cfr.py` records, at every checkpoint, the exact NashGap (for
validation only) and the certificates of the paper:

- `certificate_seq` = C^1, the profile-sequence certificate with the exact
  comparator term (one signed backward pass per player with nonzero
  comparator weight per checkpoint);
- `certificate_seq_l1` = C^0, its zero-best-response form (||w_i||_1);
- `certificate_seq_md`, the M*D relaxation;
- `certificate_seq_terms`, the per-player terms [Rhat, Delta, L, ||w||_1, M*D];
- `certificate_tight`, the unaligned Rhat+Delta sum (not a bound under
  alternation), and with `strict_regret_logging=True` the aligned
  start-of-round certificate `certificate_tight_strict`.

All accumulators are updated inside the same tree pass the update performs;
only the aligned certificate needs the extra pass.

## Full reproduction

Every stage takes `--smoke` (minutes, separate tag) so it can be exercised
before the full run; the commands below are the paper's configurations.

### 1. Biased Leduc, 7 original arms (seeds 0--63, T = 2^18, 128 checkpoints)

```
python experiments/run_m1_seq.py --workers 8       # -> results/m1_tight_seq (aligned pass on)
python experiments/run_extra_arms.py --workers 8   # -> results/m1_extra (control arms)
python experiments/analyze_m1_seq.py --no-regression   # Table 1 output rows and appendix tables
python experiments/analyze_extra_arms.py           # control-arm rows
```

`analyze_m1_seq.py` prints, for the scores `short` (unaligned sum), `strict`
(aligned), `seq` (C^1), `seq_l1` (C^0) and `seq_md`: per-arm and portfolio
selection and certification fractions at the four thresholds, soundness
counts over all 57,344 checkpoints, tightness quantiles, the size of the
comparator term, the pool ablation (`full` / `no-DPRM+` / `PRM+-core`) and the
selection-policy ablation. Its regression block re-derives the trajectories
and checks them against the `m1_tight(_strict)` reference logs. Those logs are
a legacy run directory that this package does not bundle, so the command above
passes `--no-regression`; without the flag the block skips itself and prints a
message when the directory is absent.

### 2. Held-out instances, 7 arms (seeds 64--127, same budget)

```
python experiments/run_m1_heldout.py --workers 8   # -> results/m1_heldout_seq
python experiments/analyze_m1_seq.py --dir results/m1_heldout_seq --no-regression
```

The selectors, threshold and migration band of this stage were frozen in
`experiments/PREREG_heldout.md` before the run; the C^0 and value-corrected
rows are diagnostic ablations added after it.

### 3. Pool extension: the nine added arms (pre-registered stage 2)

```
python experiments/run_arm_pool_v2.py --list       # arm names, families, configs
python experiments/run_arm_pool_v2.py --stage full --workers 8
python experiments/run_arm_pool_v2.py --stage heldout --workers 8 \
    --prereg experiments/PREREG_P1-3_HELDOUT_V2_20261001.md
```

`full` writes seeds 0--63 into `results/m1_pool_v2` at the same budget and
logging as `m1_tight_seq` (aligned pass on); `heldout` writes seeds 64--127
into `results/m1_pool_v2_heldout` with the held-out convention (no aligned
pass, like `m1_heldout_seq`) and refuses to run without the frozen v2
pre-registration file. The full stage runs the nine `FULL_ARMS` frozen at the
pilot stage (`--list` prints them, and the pilot grid of 35 configurations
from which they were picked one per family). All stages skip any log
that already exists.

### 4. Fresh instances, both pools (seeds 128--191, pre-registered stage 3)

```
python experiments/run_m1_heldout.py --seed-start 128 --n-seeds 64 \
    --out-dir results/m1_fresh_seq
python experiments/run_arm_pool_v2.py --stage fresh --workers 8 \
    --prereg experiments/PREREG_P1-3_FRESH_20261005.md
python experiments/analyze_fresh_v3.py        # the stage-3 readings
python experiments/analyze_selection_loss.py  # tab:selection-loss (all three sets)
```

Stage 3 draws a third instance set, seeds 128--191, from the same
distribution under the frozen v3 pre-registration. `run_m1_heldout.py` with
`--seed-start 128` writes the seven paper arms into `results/m1_fresh_seq`
(same runner and code as the held-out stage, a different instance range), and
`run_arm_pool_v2.py --stage fresh` writes the nine added arms into
`results/m1_pool_v2_fresh`, refusing to run without the v3 file.
`analyze_fresh_v3.py` prints the pre-registered readings into
`results/pool16_paper`; `analyze_selection_loss.py` reads the in-sample,
held-out and fresh run directories and writes `printout_selection_loss.txt`.

### 5. Three-player Leduc (T = 2^16, 64 checkpoints)

```
python experiments/run_leduc3_seq.py --no-strict --workers 6   # 6 paper arms
python experiments/run_leduc3_pool_v2.py --workers 6 --arms \
    pdcfr+_alt_a1.5_w5 hs-pcfr+_alt_g30 stprm+_alt_eta1000_w2 \
    dcfr+_alt_a1.5_w4 dcfr_alt_w2 sprm+_alt_eta10000_w2        # six added arms
python experiments/analyze_leduc3_seq.py                       # 6-arm table
python experiments/analyze_pool16_leduc3.py --no-external      # 12-arm table
```

`run_leduc3_seq.py` is a single game (no seeds); its default `--arms` is the
six paper arms. Drop `--no-strict` from either run script to log the aligned
start-of-round certificate as well (one extra tree pass per player per
iteration; the held-out stages of the paper use the same flag).

### 6. External games (7 + 9 arms x 64 instances per game, T = 2^16)

```
python experiments/run_external.py --game sheriff   --out-dir results/ext_sheriff_seq
python experiments/run_external.py --game goofspiel --out-dir results/ext_goofspiel_seq
python experiments/run_external_pool_v2.py --game sheriff   --workers 60
python experiments/run_external_pool_v2.py --game goofspiel --workers 60
python experiments/analyze_m1_seq.py --dir results/ext_sheriff_seq   --no-regression
python experiments/analyze_m1_seq.py --dir results/ext_goofspiel_seq --no-regression
python experiments/analyze_pool16_leduc3.py                # 16-arm rows per game
```

`run_external.py` walks seeds 0--63 (instance `s` fixes the game; the seven
arms share the instance). `run_external_pool_v2.py` runs the nine added arms
on the same instances into `results/ext_<game>_pool_v2`. After both stages,
`analyze_pool16_leduc3.py` prints the 16-arm rows for each game (the 16-arm
Goofspiel trajectory sensitivity noted in "Requirements" applies here).

### 7. The paper's two-player number table from all logs

```
python experiments/analyze_pool16_paper.py
```

merges `m1_tight_seq` + `m1_pool_v2` (in-sample) and `m1_heldout_seq` +
`m1_pool_v2_heldout` (held-out) and writes `results/pool16_paper/
printout_2p.txt` plus `pool16_paper_2p.json`: the 7-arm and 16-arm Table 1
rows, counts, tightness, failure sets, selection distributions, Wilson
intervals and McNemar tests, the held-out stage-2 table, the per-arm and pool
ablations, the budget block, and the single-arm oracle rows. It ends with a
self-check that re-derives the paper's frozen 7-arm numbers from the logs and
exits nonzero if any of the 30 checks fails.

### Quick sanity passes (minutes)

```
python3 -m pytest tests
python experiments/run_m1_seq.py --smoke
python experiments/run_m1_heldout.py --smoke
python experiments/run_arm_pool_v2.py --smoke
python experiments/run_leduc3_pool_v2.py --arms pdcfr+_alt_a1.5_w5 --smoke
python experiments/run_external.py --game sheriff --pilot
python experiments/run_external_pool_v2.py --game sheriff --smoke
```

Smoke runs write to `*_smoke` / `*_pilot` tags and are not the reported numbers.

## Expected outputs (threshold 1e-5 unless stated)

Biased Leduc, in-sample (seeds 0--63), from `pool16_paper/printout_2p.txt`:

| Quantity | 7 original arms | 16 arms |
|---|---|---|
| oracle union | 0.84 (54/64) | 0.89 (57/64) |
| selection by C^1 | 0.84 (54/64) | 0.89 (57/64) |
| selection by aligned (exact gap at pick) | 0.80 (51/64) | 0.89 (57/64) |
| certified C^1 | 0.75 (48/64) | 0.89 (57/64) |
| certified C^0 | 0.73 (47/64) | 0.89 (57/64) |
| certified aligned | 0.69 (44/64) | 0.86 (55/64) |
| regret-only selection | 0.83 (53/64) | 0.77 (49/64) |
| final average | 0.78 (50/64) | 0.88 (56/64) |
| uniform random checkpoint | 0.41 | 0.35 |
| oracle union at 1e-2 / 1e-3 / 1e-4 / 1e-5 | 1.00 / 1.00 / 0.91 / 0.84 | 1.00 / 1.00 / 0.97 / 0.89 |
| C^1 soundness violations | 0/57344 | 0/131072 |
| median certificate/gap, all checkpoints: C^1 / C^0 / aligned | 1.34 / 1.47 / 1.75 | 1.27 / 1.38 / 1.37 |
| median certificate/gap at selected: C^1 / aligned | 1.8 / 2.7 | 1.4 / 1.9 |
| C^1 tightest share of quantile levels | 98.5% | 98.5% |
| C^1 <= C^0 share of checkpoints | 100.0% | 100.0% |
| \|L2\|/gap median, p90 | 0.04x, 0.25x | 0.02x, 0.31x |
| \|\|w2\|\|_1/gap median | 0.38x | 0.27x |
| unsolved seeds | 5, 8, 24, 29, 34, 35, 50, 53, 60, 61 | 5, 8, 24, 29, 50, 53, 61 |
| median beta, unsolved vs all | 0.16 vs 0.28 | 0.17 vs 0.28 |
| selections landing on a new arm | 0/64 | 59/64 |
| candidate snapshots | 896 (7x128) | 2048 (16x128) |
| full grid: core-h / runs | 20.0 / 448 (24.4 with aligned pass) | 32.8 / 1024 (37.2 with aligned pass) |

Held-out instances (seeds 64--127, paired in-sample / held-out):

| Quantity | 7 arms in / held-out | 16 arms in / held-out |
|---|---|---|
| C^1 selection | 54/64 / 56/64 | 57/64 / 59/64 |
| C^0 selection | 54/64 / 56/64 | 57/64 / 59/64 |
| value-corrected selection | 54/64 / 56/64 | 57/64 / 59/64 |
| exact-gap selection (same grid) | 54/64 / 56/64 | 57/64 / 59/64 |
| certified C^1 | 48/64 / 55/64 | 57/64 / 59/64 |
| certified C^0 | 47/64 / 54/64 | 57/64 / 58/64 |
| regret-only selection | 53/64 / 54/64 | 49/64 / 56/64 |
| final average | 50/64 / 54/64 | 56/64 / 58/64 |
| uniform random checkpoint | 0.41 / 0.42 | 0.35 / 0.35 |
| Wilson 95% interval, C^1 selection | [0.74, 0.91] / [0.77, 0.94] | [0.79, 0.95] / [0.83, 0.97] |
| unsolved seeds | 10 seeds listed below / 65, 70, 76, 77, 85, 90, 124, 126 | 7 seeds listed above / 76, 77, 85, 124, 126 |
| selections landing on a new arm | 0/64 / 0/64 | 59/64 / 56/64 |

The 7-arm in-sample unsolved seeds are 5, 8, 24, 29, 34, 35, 50, 53, 60, 61;
with regret-only selection, seed 33 joins them. The stage-1
pre-registration (`PREREG_heldout.md`) fixed the C^1 and exact-gap selectors,
the 1e-5 threshold and the band [0.74, 0.91]; the stage-2 pre-registration
(`PREREG_P1-3_HELDOUT_V2_20261001.md`) fixed the extended pool and the band
[0.81, 0.97] = 0.890625 +/- 1.96*se. Paired increment, C^1 selection, 16 vs 7
arms: 3 solved by 16 only and 0 by 7 only, in-sample and held-out (McNemar
exact p = 0.250); for certified C^1, 9 vs 0 in-sample (p = 0.004) and 4 vs 0
held-out (p = 0.125). C^1 selection and exact-gap selection agree on the
solved-or-not decision on 64/64 instances, in every row of the table.

Fresh instances (seeds 128--191, stage 3), from `printout_fresh_v3.txt`:

| Quantity | 7 solvers | 16 solvers |
|---|---|---|
| C^1 selection | 55/64 | 57/64 [0.81, 0.97] |
| C^0 selection | 54/64 | 57/64 |
| value-corrected selection | 55/64 | 57/64 |
| exact-gap selection (same grid) | 55/64 | 57/64 |
| certified C^1 | 54/64 | 57/64 |
| certified C^0 | 54/64 | 57/64 |
| regret-only selection | 55/64 | 53/64 |
| final average | 54/64 | 57/64 |
| uniform random checkpoint | 0.43 | 0.36 |

The added solvers take 59 of the 64 picks (floor 55); the paired 16-vs-7
increment is n+ = 2, n- = 0 (McNemar exact p = 0.5). The unsolved
sixteen-solver seeds are 144, 146, 150, 163, 170, 181, 184, a subset of the
seven-solver pool's 129, 142, 144, 146, 150, 163, 170, 181, 184; across the
three instance sets the sixteen-solver C^1 fraction is 0.891, 0.922 and
0.891 (in-sample, held-out, fresh).

Selection loss (tab:selection-loss, from `printout_selection_loss.txt`):
selected gap over the exact trajectory minimum, median / 90th pct. / max
over the instances whose minimum is above 1e-12 (excluded counts in
parentheses): 1.01 / 1.36 / 42.45 (28) in-sample, 1.01 / 2.67 / 33.11 (26)
held-out, 1.00 / 1.74 / 17.07 (25) fresh.

Per-arm oracle fractions at 1e-5 (in-sample). Paper arms: RM+ p=1,2: 0.00;
PRM+ p=1: 0.67; PRM+ p=2: 0.70; DPRM+ p=1 lam=0.3: 0.73; DPRM+ p=2 lam=0.1:
0.67; DPRM+ p=2 lam=0.3: 0.75; within-arm C^1 selection equals the oracle
fraction for every arm. Added arms (oracle / C^1 selection / certified): DCFR
0.00 / 0.00 / 0.00; DCFR+ 0.00 / 0.00 / 0.00; PDCFR+ 0.66 / 0.66 / 0.66;
HS-PCFR+ 0.72 / 0.70 / 0.66; HS-DCFR 0.00 / 0.00 / 0.00; Smooth PRM+ 0.73 /
0.73 / 0.69; Stable PRM+ 0.80 / 0.80 / 0.78; ExRM+ 0.27 / 0.25 / 0.23;
IR-PRM+ 0.19 / 0.19 / 0.19.

Pool ablation (selection C^1 / certified C^1 / certified C^0 / aligned /
oracle union): 7 original arms 54/64, 48/64, 47/64, 51/64, 54/64; +
discounting family 54/64, 49/64, 49/64, 52/64, 54/64; + schedules family
54/64, 49/64, 47/64, 52/64, 54/64; + stabilised RM+ family 56/64, 56/64,
56/64, 56/64, 56/64; + IR-PRM+ 55/64, 52/64, 51/64, 54/64, 55/64; all 16
arms 57/64, 57/64, 57/64, 57/64, 57/64.

Budget (16-arm in-sample grid): 9 added arms 12.8 core-h over 576 runs;
16-arm grid 32.8 core-h over 1024 runs; 2048 candidate snapshots. Median
wall-clock per run: DCFR 55 s, DCFR+ 54 s, PDCFR+ 55 s, HS-PCFR+ 54 s,
HS-DCFR 55 s, Smooth PRM+ 60 s, Stable PRM+ 60 s, ExRM+ 161 s, IR-PRM+ 134 s.

Control arms (`analyze_extra_arms.py`, in-sample): RM+ uniform
1.00/0.33/0.00/0.00; PRM+ uniform 1.00/1.00/0.77/0.00; PRM+ p=2
simultaneous 1.00/0.73/0.64/0.41 (oracle at the four thresholds).

External games (T = 2^16, 64 paired instances), from `printout_3p.txt`:
Sheriff, C^1 selection / certified / oracle union 1.00 / 1.00 / 1.00 (64/64)
for the 7-arm and the 16-arm pool; C^1 soundness 0/28672 (7 arms) and
0/65536 (16 arms); the 16-arm selection lands on HS-PCFR+ (33) or PDCFR+ (31)
on every instance. Goofspiel: 7 arms 0.56 / 0.45 / 0.58 (36/64, 29/64, 37/64),
best single paper arm 0.50 (DPRM+ p=2 lam=0.1); 16 arms 0.62 / 0.58 / 0.64
(40/64, 37/64, 41/64), best single added arm 0.53 (Smooth PRM+); median
C^1/gap at the selected checkpoint 8.28x (7 arms) and 2.03x over the 51
instances with selected gap >= 1e-12 (16 arms); soundness 0/28672 and
0/65536. Goofspiel DPRM+ trajectories are the ones sensitive to last-bit
floating-point differences across NumPy builds (see "Requirements").

Three-player Leduc (12 arms, T = 2^16, 64 checkpoints): every arm's
soundness is 0/192. Trajectory minima: PDCFR+ 1.17e-06 (t = 59392), HS-PCFR+
2.16e-06 (t = 65536), PRM+ p=2 5.50e-06, RM+ p=2 1.06e-04; 8 of 12 arms
attain their minimum at T. The cross-arm C^1 argmin selects HS-PCFR+ with
certificate 2.83e-06 (1.31x its own gap), whose gap is 1.84x the pool's
trajectory minimum; the reported NPCFR+ value 1e-3 is 354x that certificate
and 853x the trajectory minimum (131x with the six paper arms only). The
per-arm table is in the paper's appendix.

## License

The code and the bundled audit logs are released under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/); see
`LICENSE`. Copyright (c) 2026 Boning Li and Longbo Huang.
