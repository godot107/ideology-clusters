"""What dimension 2 means in the 119th House: which roll calls separate members along it.

For every roll call where at least 5% of a party broke with that party's majority, correlate "voted
with my party" with each member's dimension-1 and dimension-2 position (evaluate.breakaway), then
summarise by kind of vote. Needs scripts/run_analysis.py first (results/members_119H.csv).

    python scripts/dimension2.py      # -> results/dim2_rollcalls_119H.csv, dim2_summary_119H.csv, dim2_119H.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ideology import evaluate, matrix, voteview

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    mem = pd.read_csv(RES / "members_119H.csv").set_index("icpsr")
    m = matrix.build(119, "H").loc[mem.index]
    rc = voteview.rollcalls(119, "H")
    b = evaluate.breakaway(m, mem[["dim1", "dim2"]], mem.party)
    b = b.join(rc[["date", "bill_number", "vote_question", "vote_desc"]], on="rollnumber")
    b["type"] = b.vote_question.map(voteview.vote_type)
    b.to_csv(RES / "dim2_rollcalls_119H.csv", index=False)

    summ = (b.groupby(["party", "type"])
            .agg(roll_calls=("rollnumber", "size"), dissent=("dissent", "mean"), party_yea=("party_yea", "mean"),
                 r_dim1=("r_dim1", "mean"), r_dim2=("r_dim2", "mean")).reset_index())
    total = b.groupby("party").agg(roll_calls=("rollnumber", "size"), dissent=("dissent", "mean"),
                                   party_yea=("party_yea", "mean"), r_dim1=("r_dim1", "mean"),
                                   r_dim2=("r_dim2", "mean")).reset_index().assign(type="all")
    summ = pd.concat([summ, total], ignore_index=True)
    summ.to_csv(RES / "dim2_summary_119H.csv", index=False)

    # Freedom Caucus vs other Republicans on suspension bills most Republicans backed
    rep = mem[mem.party == "Republican"]
    fc = rep.caucus.str.contains("Freedom Caucus")
    susp = [r for r in m.columns if voteview.vote_type(rc.loc[r, "vote_question"]) == "suspension"
            and m.loc[rep.index, r].mean() >= 0.5]
    nay = 1 - m.loc[rep.index, susp]                     # 1 = voted no; NaN = didn't vote
    share_no = nay.mean(axis=1)
    # One reading for both parties: yes-rate on votes where both party majorities voted yes
    dem = mem[mem.party == "Democrat"]
    both = [r for r in m.columns if m.loc[dem.index, r].mean() >= 0.5 and m.loc[rep.index, r].mean() >= 0.5]
    yes = m[both].mean(axis=1)
    biyes, biyes1 = {}, {}
    for name, g in (("democrats", dem), ("republicans", rep)):
        biyes[name] = float(g.dim2.corr(yes[g.index], method="spearman"))
        biyes1[name] = float(g.dim1.corr(yes[g.index], method="spearman"))
    caucus_rates = {}
    for name, g, cau in (("freedom_caucus", rep, "Freedom Caucus"), ("progressive_caucus", dem, "Progressive Caucus")):
        inc = g.caucus.str.contains(cau)
        caucus_rates[f"{name}_median_bipartisan_yes"] = float(yes[g.index][inc].median())
        caucus_rates[f"{name}_rest_of_party_median_bipartisan_yes"] = float(yes[g.index][~inc].median())
        caucus_rates[f"{name}_median_dim2"] = float(g.dim2[inc].median())
        caucus_rates[f"{name}_rest_of_party_median_dim2"] = float(g.dim2[~inc].median())

    out = {
        "suspension_bills_most_republicans_backed": len(susp),
        "freedom_caucus_members": int(fc.sum()),
        "freedom_caucus_median_no_share": float(share_no[fc].median()),
        "other_republicans_median_no_share": float(share_no[~fc].median()),
        "freedom_caucus_median_dim2": float(rep.dim2[fc].median()),
        "other_republicans_median_dim2": float(rep.dim2[~fc].median()),
        "republicans_dim2_vs_no_share_spearman": float(rep.dim2.corr(share_no, method="spearman")),
        "bipartisan_votes_both_majorities_yes": len(both),
        **{f"{k}_dim2_vs_bipartisan_yes_spearman": v for k, v in biyes.items()},
        **{f"{k}_dim1_vs_bipartisan_yes_spearman": v for k, v in biyes1.items()},
        **caucus_rates,
        "rollcall_text_note": "Voteview has no amendment text for these roll calls (dtl_desc empty).",
    }
    (RES / "dim2_119H.json").write_text(json.dumps(out, indent=2) + "\n")
    print(summ.round(2).to_string(index=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
