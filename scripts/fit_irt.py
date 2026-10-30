"""Fit the Bayesian ideal-point model and save posterior draws (data/irt_<congress><chamber>.npy)
plus diagnostics and per-member positions (results/irt_<congress><chamber>_{diagnostics.json,positions.csv}). ~5-15 min each on CPU.

    .venv/bin/python scripts/fit_irt.py 118S 118H 119H
    .venv/bin/python scripts/fit_irt.py 119H --chains 2 --warmup 1000   # longer run, real R-hat
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from ideology import embed, irt, matrix

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("tags", nargs="*", default=["118S", "118H", "119H"])
ap.add_argument("--chains", type=int, default=1)
ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--samples", type=int, default=500)
args = ap.parse_args()

for tag in args.tags:
    congress, chamber = int(tag[:-1]), tag[-1]
    m = matrix.build(congress, chamber)
    info = matrix.with_members(m, congress, chamber)
    right = (info.party == "Republican").to_numpy()
    z, _ = embed.lpca(m, 1, right_mask=right)
    t0 = time.time()
    x, diag = irt.fit(m, right, warmup=args.warmup, samples=args.samples, init_x=z[:, 0], chains=args.chains)
    diag.update(chains=args.chains, warmup=args.warmup, samples=args.samples)
    diag["seconds"] = round(time.time() - t0)
    np.save(ROOT / "data" / f"irt_{tag}.npy", x)
    irt.positions_table(x, m, info).to_csv(ROOT / "results" / f"irt_{tag}_positions.csv")
    (ROOT / "results" / f"irt_{tag}_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    print(tag, diag, flush=True)
