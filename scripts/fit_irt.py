"""Fit the Bayesian ideal-point model and save posterior draws (data/irt_<congress><chamber>.npy)
plus diagnostics (results/irt_<congress><chamber>_diagnostics.json). ~5-15 min each on CPU.

    .venv/bin/python scripts/fit_irt.py 118S 118H 119H
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from ideology import embed, irt, matrix

ROOT = Path(__file__).resolve().parents[1]

for tag in sys.argv[1:] or ["118S", "118H", "119H"]:
    congress, chamber = int(tag[:-1]), tag[-1]
    m = matrix.build(congress, chamber)
    info = matrix.with_members(m, congress, chamber)
    right = (info.party == "Republican").to_numpy()
    z, _ = embed.lpca(m, 1, right_mask=right)
    t0 = time.time()
    x, diag = irt.fit(m, right, warmup=500, samples=500, init_x=z[:, 0])
    diag["seconds"] = round(time.time() - t0)
    np.save(ROOT / "data" / f"irt_{tag}.npy", x)
    (ROOT / "results" / f"irt_{tag}_diagnostics.json").write_text(json.dumps(diag, indent=2) + "\n")
    print(tag, diag, flush=True)
