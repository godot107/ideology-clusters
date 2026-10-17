"""The vote matrix every method starts from: one row per legislator, one column per roll call,
1 = yea, 0 = nay, NaN = didn't vote (or wasn't serving yet).

Filters follow the NOMINATE convention so comparisons with DW-NOMINATE are fair:
- drop *lopsided* roll calls where the losing side has under 2.5% of votes: near-unanimous votes
  (naming a post office) carry almost no information about ideology;
- drop legislators with fewer than 20 recorded votes on the remaining roll calls.
"""
import numpy as np
import pandas as pd

from ideology import voteview

MIN_MINORITY = 0.025
MIN_VOTES = 20


def build(congress, chamber):
    v = voteview.votes(congress, chamber)
    # The votes file also records the President's announced positions as if he were a member
    # (e.g. icpsr 99913 in the 118th): keep only members of this chamber.
    v = v[v.icpsr.isin(voteview.members(congress, chamber).index)]
    v = v[v.cast_code.isin(voteview.YEA | voteview.NAY)]
    v = v.assign(y=v.cast_code.isin(voteview.YEA).astype(float))
    m = v.pivot_table(index="icpsr", columns="rollnumber", values="y", aggfunc="first")
    share = m.mean()
    m = m.loc[:, (share >= MIN_MINORITY) & (share <= 1 - MIN_MINORITY)]
    m = m[m.notna().sum(axis=1) >= MIN_VOTES]
    return m


def with_members(m, congress, chamber):
    """Member info (name, party, state, DW-NOMINATE) aligned to the matrix rows."""
    info = voteview.members(congress, chamber)
    info = info[~info.index.duplicated()]
    return info.reindex(m.index)


def summary(m):
    return {"legislators": m.shape[0], "roll_calls": m.shape[1],
            "missing_share": float(m.isna().mean().mean()),
            "yea_share": float(np.nanmean(m.to_numpy()))}


def impute_column_mean(m):
    """Simplest missing-vote fill, for PCA only: a missed vote becomes that roll call's yea share.
    Biased toward the middle for frequent absentees; later methods model missingness instead."""
    return m.fillna(m.mean())


