"""Pilot regression: on biased Shapley, DPRM+ must reach NashGap < 1e-2
while RM+ stays stuck near the correlated CCE (gap > 5e-2)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CFRConfig, build_efg, run_cfr
from src.games import make_biased_shapley


def main() -> None:
    efg = build_efg(make_biased_shapley(0.25))
    t_max = 1 << 13
    base = dict(t_max=t_max, weight_exp=2.0, eval_every=t_max)

    out_rm = run_cfr(efg, CFRConfig(alg="rm+", **base))
    gap_rm = out_rm["final"]["nash_gap"]
    kl_rm = out_rm["final"]["corr_kl"]

    out_d = run_cfr(efg, CFRConfig(
        alg="dprm+", lam=1e3, clip_c0=8.0, clip_alpha=0.25, **base))
    gap_d = out_d["final"]["nash_gap"]
    kl_d = out_d["final"]["corr_kl"]

    print(f"biased_shapley(eta=0.25), T={t_max}:")
    print(f"  rm+  : gap={gap_rm:.3e}, kl={kl_rm:.2e}")
    print(f"  dprm+: gap={gap_d:.3e}, kl={kl_d:.2e}")
    assert gap_rm > 5e-2, f"rm+ unexpectedly solved it: {gap_rm}"
    assert gap_d < 1e-2, f"dprm+ failed to decorrelate: {gap_d}"
    assert kl_d < kl_rm / 100.0, (kl_d, kl_rm)
    print("ALL PASSED: test_dprm_matrix")


if __name__ == "__main__":
    main()
