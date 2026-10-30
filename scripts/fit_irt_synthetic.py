"""Bayesian ideal points on the synthetic Congress, 1-D vs 2-D: do the 90% ranges contain the true
positions 90% of the time, per faction? The hard-line faction differs on dimension 2, so a 1-D model
should miss it and a 2-D model shouldn't. ~40 min on CPU (2 chains each, run in parallel).

Before scoring, draws are put on a common footing, which changes nobody's position relative to anyone
else: 1-D draws are standardised (irt.standardize; the axis's shift and stretch are only weakly pinned),
2-D draws are rotated onto the logistic-PCA map (irt.align; the rotation is only weakly pinned).
Diagnostics are reported raw and after that step.

    python scripts/fit_irt_synthetic.py              # fit + score -> data/sim_irt{,2d}_x.npy,
                                                     #   results/irt_synthetic_{coverage.csv,diagnostics.json}
    python scripts/fit_irt_synthetic.py --rescore    # score the saved draws again (seconds)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ideology import embed, irt, simulate

ROOT = Path(__file__).resolve().parents[1]
DRAWS1, DRAWS2 = ROOT / "data" / "sim_irt_x.npy", ROOT / "data" / "sim_irt2d_x.npy"
DIAG = ROOT / "results" / "irt_synthetic_diagnostics.json"
CHAINS = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="skip fitting; score the saved draws")
    args = ap.parse_args()

    c = simulate.simulate()
    right = (c.members.party == "R").to_numpy()
    z1, _ = embed.lpca(c.votes, 1, right_mask=right)
    z2, _ = embed.lpca(c.votes, 2, right_mask=right)
    diags = json.loads(DIAG.read_text()) if args.rescore and DIAG.exists() else {}

    if not args.rescore:
        t0 = time.time()
        x1, d = irt.fit(c.votes, right, warmup=500, samples=500, init_x=z1[:, 0], chains=CHAINS)
        diags["1-D"] = {"raw": d, "seconds": round(time.time() - t0)}
        np.save(DRAWS1, x1)
        print("1-D", diags["1-D"], flush=True)
        t0 = time.time()
        x2, d = irt.fit2d(c.votes, c.members.party, z2, warmup=1000, samples=500, chains=CHAINS)
        diags["2-D"] = {"raw": d, "seconds": round(time.time() - t0)}
        np.save(DRAWS2, x2)
        print("2-D", diags["2-D"], flush=True)

    x1 = irt.standardize(np.load(DRAWS1))
    x2 = irt.align(np.load(DRAWS2), z2)
    diags.setdefault("1-D", {})["standardized"] = irt.chain_diagnostics(x1, CHAINS)
    diags.setdefault("2-D", {})["aligned"] = irt.chain_diagnostics(x2, CHAINS)

    cov = pd.DataFrame({"1-D model": irt.coverage(x1, c.members.x1, c.members.faction)})
    cov2 = irt.coverage2d(x2, c.members[["x1", "x2"]].to_numpy(), c.members.faction)
    cov["2-D model, dim 1"], cov["2-D model, dim 2"] = cov2["dim 1"], cov2["dim 2"]
    cov = cov.round(3)
    cov.index.name = "faction"
    cov.to_csv(ROOT / "results" / "irt_synthetic_coverage.csv")
    DIAG.write_text(json.dumps(diags, indent=2) + "\n")
    print(cov.to_string())
    print(json.dumps({k: {kk: v for kk, v in d.items() if kk != "raw"} for k, d in diags.items()}, indent=2))


if __name__ == "__main__":
    main()
