"""2-D Bayesian ideal points on the real 119th House: error bars on both dimensions, so we can say how
sure we are that the Freedom Caucus sits apart on dimension 2 (the yes/no-on-bipartisan-bills axis).
~1 h on CPU (2 chains in parallel). Needs scripts/run_analysis.py and scripts/dimension2.py first.

Draws are rotated onto the logistic-PCA map before scoring (irt.align): the model only weakly pins the
rotation, and alignment changes nobody's position relative to anyone else. The reference is logistic
PCA, never DW-NOMINATE.

    python scripts/fit_irt2d.py            # fit + score -> data/irt2d_119H.npy,
                                           #   results/irt2d_119H_{positions.csv,summary.json}
    python scripts/fit_irt2d.py --rescore  # score the saved draws again (seconds)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ideology import irt, matrix

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DRAWS = ROOT / "data" / "irt2d_119H.npy"
SUMMARY = RES / "irt2d_119H_summary.json"
CHAINS = 2


def spearman(a, b):
    return float(pd.Series(np.asarray(a)).corr(pd.Series(np.asarray(b)), method="spearman"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", action="store_true", help="skip fitting; score the saved draws")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=500)
    args = ap.parse_args()

    mem = pd.read_csv(RES / "members_119H.csv").set_index("icpsr")
    m = matrix.build(119, "H").loc[mem.index]
    ref = mem[["dim1", "dim2"]].to_numpy()               # logistic PCA, 2 dims (run_analysis.py)
    old = json.loads(SUMMARY.read_text()) if args.rescore and SUMMARY.exists() else {}

    if args.rescore:
        diag = old.get("raw_diagnostics", {})
    else:
        t0 = time.time()
        x, diag = irt.fit2d(m, mem.party.to_numpy(), ref, warmup=args.warmup, samples=args.samples,
                            chains=CHAINS)
        diag = {**diag, "chains": CHAINS, "warmup": args.warmup, "samples": args.samples,
                "seconds": round(time.time() - t0)}
        np.save(DRAWS, x)
        print("raw", diag, flush=True)

    x = irt.align(np.load(DRAWS), ref)
    lo, hi = np.percentile(x, 5, axis=0), np.percentile(x, 95, axis=0)
    mean = x.mean(0)
    pos = pd.DataFrame({"name": mem.name, "party": mem.party, "state": mem.state, "district": mem.district,
                        "caucus": mem.caucus,
                        "dim1": mean[:, 0], "dim1_lo": lo[:, 0], "dim1_hi": hi[:, 0],
                        "dim2": mean[:, 1], "dim2_lo": lo[:, 1], "dim2_hi": hi[:, 1]}, index=mem.index)
    pos.round(4).to_csv(RES / "irt2d_119H_positions.csv")

    yes = pd.read_csv(RES / "dim2_members_119H.csv").set_index("icpsr").bipartisan_yes.loc[mem.index]
    one_d = pd.read_csv(RES / "irt_119H_positions.csv").set_index("icpsr")
    out = {"raw_diagnostics": diag, "aligned_diagnostics": irt.chain_diagnostics(x, CHAINS),
           "members": len(mem), "roll_calls": m.shape[1]}
    for party, key in (("Republican", "republicans"), ("Democrat", "democrats")):
        g = mem.party == party
        out[f"{key}_dim1_vs_nominate_dim1_spearman"] = spearman(pos.dim1[g], mem.nominate_dim1[g])
        out[f"{key}_dim2_vs_bipartisan_yes_spearman"] = spearman(pos.dim2[g], yes[g])
        out[f"{key}_dim2_vs_lpca_dim2_spearman"] = spearman(pos.dim2[g], mem.dim2[g])

    rep = pos[pos.party == "Republican"]
    fc = rep.caucus.str.contains("Freedom Caucus")
    rest_median = float(rep.dim2[~fc].median())
    # Is a member's dim-2 range clear of the median non-caucus Republican?  Sign: dim 2 runs the same
    # way as logistic PCA dim 2, so "apart" is whichever side the caucus median falls on.
    side = np.sign(rep.dim2[fc].median() - rest_median)
    clear = (rep.dim2_hi < rest_median) if side < 0 else (rep.dim2_lo > rest_median)
    w1 = (rep.dim1_hi - rep.dim1_lo)
    w1_1d = one_d.loc[rep.index, "width"] / one_d.loc[one_d.party == "Republican", "position"].std()
    w1_2d = w1 / pos.dim1[pos.party == "Republican"].std()
    out.update({
        "freedom_caucus_members": int(fc.sum()),
        "freedom_caucus_median_dim2": float(rep.dim2[fc].median()),
        "other_republicans_median_dim2": rest_median,
        "freedom_caucus_median_dim2_width": float((rep.dim2_hi - rep.dim2_lo)[fc].median()),
        "other_republicans_median_dim2_width": float((rep.dim2_hi - rep.dim2_lo)[~fc].median()),
        "freedom_caucus_share_dim2_range_clear_of_other_R_median": float(clear[fc].mean()),
        "other_republicans_share_dim2_range_clear_of_other_R_median": float(clear[~fc].mean()),
        # dim-1 range width in units of the Republican spread, 1-D vs 2-D model (scale-free comparison)
        "freedom_caucus_median_dim1_width_1d_model_in_R_sd": float(w1_1d[fc].median()),
        "freedom_caucus_median_dim1_width_2d_model_in_R_sd": float(w1_2d[fc].median()),
        "other_republicans_median_dim1_width_1d_model_in_R_sd": float(w1_1d[~fc].median()),
        "other_republicans_median_dim1_width_2d_model_in_R_sd": float(w1_2d[~fc].median()),
    })
    SUMMARY.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
