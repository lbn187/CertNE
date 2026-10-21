"""RM+ on Kuhn poker must converge: NashGap < 1e-2 and value ~ -1/18."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import CFRConfig, build_efg, run_cfr, utilities
from src.games import PokerGame, kuhn_rules


def main() -> None:
    efg = build_efg(PokerGame(kuhn_rules(2)))
    cfg = CFRConfig(alg="rm+", t_max=10000, weight_exp=1.0, eval_every=2000)
    out = run_cfr(efg, cfg)
    gap = out["final"]["nash_gap"]
    value = utilities(efg, out["xs_bar"])[0]
    print(f"kuhn2 rm+: NashGap={gap:.2e}, value(P0)={value:.5f} "
          f"(target -1/18 = {-1/18:.5f}), wall={out['wall_seconds']:.1f}s")
    assert gap < 1e-2, gap
    assert abs(value - (-1.0 / 18.0)) < 0.01, value
    print("ALL PASSED: test_cfr_kuhn")


if __name__ == "__main__":
    main()
