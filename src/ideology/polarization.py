"""Polarisation per Congress, 1973 (93rd) to now (119th).

Positions fitted separately for each Congress have an arbitrary scale, so only scale-free measures are
compared across time:
- separation: gap between the party medians on dim 1, divided by the average within-party spread
  (an effect size: how many "typical party widths" apart the parties sit);
- overlap: share of members closer to the *other* party's median than their own;
- party-line votes: share of contested roll calls where most Democrats and most Republicans voted
  opposite ways. Straight from the raw votes, no model at all.
The same separation and overlap are computed from DW-NOMINATE (the known answer) for comparison.
"""
import numpy as np
import pandas as pd

from ideology import embed, matrix


def _sep_overlap(x, party):
    d, r = x[party == "Democrat"], x[party == "Republican"]
    md, mr = np.median(d), np.median(r)
    spread = (np.std(d) + np.std(r)) / 2
    closer_other = np.r_[np.abs(d - mr) < np.abs(d - md), np.abs(r - md) < np.abs(r - mr)]
    return abs(mr - md) / spread, float(closer_other.mean())


def party_line_share(votes, party):
    d = votes[party == "Democrat"].mean()
    r = votes[party == "Republican"].mean()
    return float((((d > 0.5) & (r < 0.5)) | ((d < 0.5) & (r > 0.5))).mean())


def one(congress, chamber):
    m = matrix.build(congress, chamber)
    info = matrix.with_members(m, congress, chamber)
    party = info.party.to_numpy()
    keep = np.isin(party, ["Democrat", "Republican"])
    z, _ = embed.lpca(m, 1, lam=1.0, right_mask=party == "Republican")
    sep, ov = _sep_overlap(z[keep, 0], party[keep])
    nom = info.nominate_dim1.to_numpy()
    ok = keep & ~np.isnan(nom)
    nsep, nov = _sep_overlap(nom[ok], party[ok])
    return {"congress": congress, "chamber": chamber, "start_year": 1789 + 2 * (congress - 1),
            "members": int(keep.sum()), "roll_calls": m.shape[1],
            "separation": sep, "overlap": ov, "nominate_separation": nsep, "nominate_overlap": nov,
            "party_line_share": party_line_share(m[keep], party[keep])}


def series(congresses=range(93, 120), chambers=("H", "S")):
    return pd.DataFrame([one(c, ch) for ch in chambers for c in congresses])
