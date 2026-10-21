# Side experiment: certificate-gated reference homotopy (Leduc)

The two claims in the paper's Future Work section come from using the
certificate to *control* a solver; the rest of the paper uses it to select
from a portfolio. Both claims come from a regularized entropy-descent solver
on Leduc, 5,000 iterations, that advances a fixed-average reference every 200
iterations.

| Claim | Evidence |
|---|---|
| The aligned certificate stayed sound: 13 anchor updates, final anchor NashConv 0.0139 against its stored bound 0.0205 | `stage3b_dual_leduc_5000_summary.csv` row `dual_anchor_p200` (`anchor_moves=13`, `anchor_final_nc`, `anchor_bound`, `bound_violations=0`); per-move audit in `stage3b_dual_leduc_5000_audit.csv`; console log `stage3b_dual_leduc_5000.log` |
| Gating advances on certified progress froze the dynamics after four advances: final NashConv 0.109 against 0.022 ungated | same log, rows `online_tight_p200` (`moves=4`, `nc=1.091408e-01`) and `fixed_p200_avg` (`nc=2.152212e-02`); summary `stage3a_leduc_5000_summary.csv` |
| Forcing advances past the gate violated the gate's monotone-threshold invariant at every forced step | `pilot_cert_homotopy_leduc_online.log`: `online_tight_p20_force10: accepted=51 forced=18 violations=18` |

These are the raw CSV/log artifacts of the side experiment; the code that
produced them is not part of this package. The absolute output paths in the
three `.log` files were replaced by `logs/`; no numeric field was altered. `nc` is the sum-form NashConv, the
quantity the certificate bounds when a deployment certifies the profile sum.
