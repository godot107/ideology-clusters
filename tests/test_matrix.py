"""The vote-matrix filters on a tiny hand-made Congress (no download)."""
import pandas as pd

from ideology import matrix, voteview


def fake_votes(rows):
    return pd.DataFrame(rows, columns=["congress", "chamber", "rollnumber", "icpsr", "cast_code"])


def test_build_codes_and_filters(monkeypatch):
    rows = []
    for icpsr in range(1, 41):
        rows.append((118, "House", 1, icpsr, 1 if icpsr <= 20 else 6))     # split vote: kept
        rows.append((118, "House", 2, icpsr, 1))                           # unanimous: dropped
        rows.append((118, "House", 3, icpsr, 9 if icpsr == 1 else 2))      # not voting -> NaN
        for r in range(4, 24):                                             # 20 more split votes
            rows.append((118, "House", r, icpsr, 1 if (icpsr + r) % 2 else 4))
    rows.append((118, "House", 1, 99, 1))                                  # member with 1 vote: dropped
    for r in range(1, 24):
        rows.append((118, "House", r, 99913, 1))                           # the President: dropped
    monkeypatch.setattr(voteview, "votes", lambda c, ch: fake_votes(rows))
    members = pd.DataFrame(index=pd.Index(list(range(1, 41)) + [99], name="icpsr"))
    monkeypatch.setattr(voteview, "members", lambda c, ch: members)
    m = matrix.build(118, "H")
    assert 2 not in m.columns and 3 not in m.columns    # roll call 3 is 39-0 among voters: lopsided
    assert 99 not in m.index and 99913 not in m.index
    assert m.loc[1, 1] == 1.0 and m.loc[21, 1] == 0.0
    assert m.shape == (40, 21)
