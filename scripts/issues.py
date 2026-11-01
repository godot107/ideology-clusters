"""Issues and individuals in the 119th House: which policy areas split the parties, which bring them
together, and how often each member votes against their own party. Needs scripts/fetch_bills.py 119
(Congress.gov topics) and scripts/run_analysis.py (positions + caucus labels).

Unlike the ideology stages, this uses *every* roll call (no lopsided-vote filter): near-unanimous votes
are exactly the "both parties agree" cases.

Definitions (all from raw votes, no model):
- yes-rate: share of a party's voting members who voted yes (any kind of yea) on a roll call;
- gap: |D yes-rate - R yes-rate| (0 = parties voted alike, 1 = perfect party line);
- bipartisan: gap < 0.25; party-line: gap > 0.8;
- party-split vote: the two parties' majorities voted opposite ways (procedural rule votes excluded);
- against own party: share of a member's party-split votes cast with the other party's majority;
- missed votes: share of all roll calls held while the member was serving where they didn't vote.

    python scripts/issues.py   # -> results/issues_{rollcalls,policy_areas,members}_119H.csv, issues_119H.json
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ideology import congress_gov, voteview

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
C, CH = 119, "H"
SUBSTANTIVE = ("passage", "suspension", "amendment")
TERRITORIES = ("GU", "PR", "VI", "AS", "MP", "DC")   # delegates: vote on amendments only


def main():
    mem = voteview.members(C, CH)
    mem = mem[~mem.index.duplicated()]
    raw = voteview.votes(C, CH)
    raw = raw[raw.icpsr.isin(mem.index)]
    cast = raw[raw.cast_code.isin(voteview.YEA | voteview.NAY)].assign(
        y=lambda v: v.cast_code.isin(voteview.YEA).astype(float))
    M = cast.pivot_table(index="icpsr", columns="rollnumber", values="y", aggfunc="first")
    serving = raw[raw.cast_code != 0].pivot_table(index="icpsr", columns="rollnumber", values="cast_code",
                                                  aggfunc="first").notna()
    party = mem.party_code.reindex(M.index).map({100: "D", 200: "R"})

    rc = voteview.rollcalls(C, CH).reindex(M.columns)
    rc["type"] = rc.vote_question.map(voteview.vote_type)
    topics = {b: congress_gov.bill(C, b) for b in rc.bill_number.dropna().unique() if congress_gov.parse_bill(b)}
    rc["policy_area"] = rc.bill_number.map(lambda b: (topics.get(b) or {}).get("policy_area"))
    rc["bill_title"] = rc.bill_number.map(lambda b: (topics.get(b) or {}).get("title"))
    rc["bill_url"] = rc.bill_number.map(lambda b: (topics.get(b) or {}).get("url"))
    d, r = M[party == "D"].mean(), M[party == "R"].mean()
    rc["D_yes"], rc["R_yes"], rc["gap"] = d, r, (d - r).abs()
    rc["D_voting"], rc["R_voting"] = M[party == "D"].notna().sum(), M[party == "R"].notna().sum()
    rc.reset_index().to_csv(RES / f"issues_rollcalls_{C}{CH}.csv", index=False)
    # every member's vote on final-passage and fast-track votes on bills (for the look-up app)
    keep = rc.index[rc.type.isin(("passage", "suspension")) & rc.bill_title.notna()]
    code = raw[raw.rollnumber.isin(keep)].assign(
        vote=lambda v: np.select([v.cast_code.isin(voteview.YEA), v.cast_code.isin(voteview.NAY)], ["Yes", "No"],
                                 "Not voting"))
    code[code.cast_code != 0][["icpsr", "rollnumber", "vote"]].to_csv(
        RES / f"issues_member_votes_{C}{CH}.csv.gz", index=False)

    # --- policy areas (substantive votes with a Congress.gov policy area)
    sub = rc[rc.type.isin(SUBSTANTIVE) & rc.policy_area.notna()]
    pa = (sub.groupby("policy_area")
          .agg(votes=("gap", "size"), bills=("bill_number", "nunique"), median_gap=("gap", "median"),
               bipartisan_share=("gap", lambda x: float((x < 0.25).mean())),
               party_line_share=("gap", lambda x: float((x > 0.8).mean())))
          .sort_values("median_gap"))
    pa.to_csv(RES / f"issues_policy_areas_{C}{CH}.csv")

    # --- members
    split = rc.index[((d > 0.5) != (r > 0.5)) & (rc.type != "rule")]
    both_yes = rc.index[(d > 0.5) & (r > 0.5)]
    pos = pd.read_csv(RES / f"members_{C}{CH}.csv").set_index("icpsr")
    rows = []
    for i in M.index:
        if party[i] not in ("D", "R"):
            continue
        x = M.loc[i, split].dropna()
        own = (d[x.index] > 0.5) if party[i] == "D" else (r[x.index] > 0.5)
        held = serving.loc[i].reindex(rc.index, fill_value=False) if i in serving.index else None
        n_held = int(held.sum()) if held is not None else 0
        n_cast = int(M.loc[i].notna().sum())
        rows.append({
            "icpsr": i, "name": mem.bioname[i], "party": party[i], "state": mem.state_abbrev[i],
            "district": int(mem.district_code[i]), "delegate": mem.state_abbrev[i] in TERRITORIES,
            "votes_cast": n_cast, "missed_share": 1 - n_cast / n_held if n_held else np.nan,
            "split_votes": len(x), "against_party_share": float((x != own.astype(float)).mean()) if len(x) else np.nan,
            "bipartisan_yes_share": float(M.loc[i, both_yes].mean()),
            "caucus": pos.caucus.get(i, "none") if i in pos.index else "none",
            "dim1": pos.dim1.get(i, np.nan) if i in pos.index else np.nan,
            "dim2": pos.dim2.get(i, np.nan) if i in pos.index else np.nan,
        })
    X = pd.DataFrame(rows).set_index("icpsr")
    # where a member sits in their own party on left-right: 0 = nearest the other party, 1 = farthest
    # (dim 1 has Republicans positive, so Democrats are farther out the more negative they are)
    for p in "DR":
        g = X.party == p
        X.loc[g, "dim1_pct_from_center"] = (-X.dim1[g] if p == "D" else X.dim1[g]).rank(pct=True)
    X.to_csv(RES / f"issues_members_{C}{CH}.csv")

    # --- data centres and other searchable themes, from Congress.gov subjects + summaries + titles
    def mentions(pat, summary=False):
        hits = []
        for b, t in topics.items():
            if not t or t.get("missing"):
                continue
            text = " ".join([t.get("title") or "", " ".join(t.get("subjects") or []),
                             (t.get("summary") or "") if summary else ""])
            if re.search(pat, text, re.I):
                hits.append(b)
        sel = rc[rc.bill_number.isin(hits) & rc.type.isin(SUBSTANTIVE)]
        return sel[["date", "bill_number", "bill_title", "vote_question", "D_yes", "R_yes", "gap"]].round(3)

    # titles + CRS subject terms only (summaries of big bills mention everything), except data centres,
    # which no subject term covers: there the summary is the only place they're named
    themes = {"data centers": (r"data cent(er|re)", True),
              "artificial intelligence": (r"artificial intelligence", False),
              "crypto / digital assets": (r"digital asset|stablecoin|crypto", False),
              "electricity prices / grid": (r"utility rates|energy prices|electric power generation", False)}
    out = {"roll_calls": len(rc), "substantive_with_policy_area": len(sub), "bills_looked_up": len(topics),
           "bills_missing_on_congress_gov": sum(1 for t in topics.values() if t and t.get("missing")),
           "party_split_votes": len(split), "both_majorities_yes_votes": len(both_yes),
           "median_against_party": {p: float(X[(X.party == p) & ~X.delegate].against_party_share.median()) for p in "DR"},
           "median_missed": {p: float(X[(X.party == p) & ~X.delegate].missed_share.median()) for p in "DR"},
           "themes": {k: mentions(*v).assign(date=lambda t: t.date.astype(str)).to_dict("records") for k, v in themes.items()}}
    (RES / f"issues_{C}{CH}.json").write_text(json.dumps(out, indent=1, default=str) + "\n")

    pd.set_option("display.width", 220)
    print(pa.round(2).to_string())
    print(json.dumps({k: v for k, v in out.items() if k != "themes"}, indent=1))
    for k, v in out["themes"].items():
        print(f"\n== {k}: {len(v)} substantive roll calls")
        for row in v[:12]:
            print(f"  {row['date']} {row['bill_number']:8} D {row['D_yes']:.2f} R {row['R_yes']:.2f}  {str(row['bill_title'])[:90]}")
    for p in "DR":
        t = X[(X.party == p) & ~X.delegate].sort_values("against_party_share", ascending=False)
        print(f"\n== {p}: most often against own party")
        print(t[["name", "state", "district", "split_votes", "against_party_share", "bipartisan_yes_share",
                 "dim1_pct_from_center", "caucus"]].head(10).round(3).to_string())


if __name__ == "__main__":
    main()
