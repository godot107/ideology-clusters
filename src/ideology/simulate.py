"""A synthetic Congress with known ideal points and known factions: the answer key every method is
tested on before it touches real votes.

Members sit in two dimensions:
- dim 1, left-right: each party has a wing, a mainstream and a moderate faction;
- dim 2, "against the leadership": one Republican faction sits high on it while being only somewhat
  further right, loosely modelled on how a hard-line bloc votes against its own party's leaders.
  It is the faction a 1-D method should struggle with.

Each roll call has a direction (mostly dim 1, some mostly dim 2), a cut line and a sharpness. A member
votes yea with probability sigmoid(sharpness * (position . direction - cut)), so votes are noisy the way
real ones are. Missing votes are NOT random: chronic absentees and members who served part of the term.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

FACTIONS = {  # name: (party, share of party, dim1 mean, dim2 mean, sd)
    "D-progressive": ("D", 0.25, -1.30, 0.00, 0.18),
    "D-mainstream": ("D", 0.60, -0.80, 0.00, 0.18),
    "D-moderate": ("D", 0.15, -0.30, 0.00, 0.15),
    "R-moderate": ("R", 0.15, 0.30, 0.00, 0.15),
    "R-mainstream": ("R", 0.65, 0.80, 0.00, 0.18),
    "R-hardline": ("R", 0.20, 1.05, 1.20, 0.18),
}


@dataclass
class Congress:
    votes: pd.DataFrame      # members x roll calls, 1 / 0 / NaN
    members: pd.DataFrame    # party, faction, x1, x2 (the truth)
    rollcalls: pd.DataFrame  # direction angle, cut, sharpness


def simulate(n_members=440, n_rollcalls=1000, dim2_share=0.15, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for party in ("D", "R"):
        for name, (p, share, m1, m2, sd) in FACTIONS.items():
            if p != party:
                continue
            for _ in range(int(round(n_members / 2 * share))):
                rows.append({"party": party, "faction": name,
                             "x1": rng.normal(m1, sd), "x2": rng.normal(m2, sd)})
    members = pd.DataFrame(rows)
    members.index = pd.RangeIndex(1, len(members) + 1, name="member")

    angle = np.where(rng.random(n_rollcalls) < dim2_share,
                     rng.normal(np.pi / 2, 0.15, n_rollcalls), rng.normal(0, 0.15, n_rollcalls))
    angle = angle + np.where(rng.random(n_rollcalls) < 0.5, 0, np.pi)   # which side is "yea"
    direction = np.c_[np.cos(angle), np.sin(angle)]
    cut = rng.uniform(-1.2, 1.2, n_rollcalls)
    sharp = rng.uniform(2.0, 6.0, n_rollcalls)
    X = members[["x1", "x2"]].to_numpy()
    p = 1 / (1 + np.exp(-sharp * (X @ direction.T - cut)))
    y = (rng.random(p.shape) < p).astype(float)

    absent = np.full(len(members), 0.03)
    chronic = rng.random(len(members)) < 0.08
    absent[chronic] = rng.uniform(0.2, 0.5, chronic.sum())
    y[rng.random(y.shape) < absent[:, None]] = np.nan
    partial = rng.choice(len(members), 12, replace=False)       # arrived late or left early
    for j, i in enumerate(partial):
        k = rng.integers(n_rollcalls // 5, n_rollcalls // 2)
        if j % 2:
            y[i, :k] = np.nan    # arrived late (special election)
        else:
            y[i, -k:] = np.nan   # left early

    votes = pd.DataFrame(y, index=members.index, columns=pd.RangeIndex(1, n_rollcalls + 1, name="rollnumber"))
    share = votes.mean()
    votes = votes.loc[:, (share >= 0.025) & (share <= 0.975)]
    rollcalls = pd.DataFrame({"angle": angle, "cut": cut, "sharpness": sharp},
                             index=pd.RangeIndex(1, n_rollcalls + 1, name="rollnumber")).loc[votes.columns]
    return Congress(votes, members, rollcalls)
