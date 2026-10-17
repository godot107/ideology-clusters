"""Every non-Bayesian result in the project, written to results/ (the notebooks read these).

    .venv/bin/python scripts/run_analysis.py [synthetic] [heldout] [clusters] [polarization]

~10 min in all on CPU (polarisation fits 54 Congress-chambers; the cluster bootstrap re-fits 20 times).
The Bayesian ideal points are separate: scripts/fit_irt.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ideology import caucus, cluster, embed, evaluate, matrix, polarization, simulate

RES = Path(__file__).resolve().parents[1] / "results"
RES.mkdir(exist_ok=True)
CAUCUS_TARGETS = [("Freedom Caucus", "Republican"), ("Problem Solvers Caucus", "Republican"),
                  ("Progressive Caucus", "Democrat"), ("Blue Dog Coalition", "Democrat"),
                  ("Problem Solvers Caucus", "Democrat")]


def synthetic():
    c = simulate.simulate()
    right = (c.members.party == "R").to_numpy()
    truth = c.members[["x1", "x2"]].to_numpy()
    rows = []
    zp = embed.pca(c.votes, 2, right)
    zl, _ = embed.lpca(c.votes, 2, right_mask=right)
    for name, z in (("PCA", zp), ("logistic PCA", zl)):
        rows.append({"method": name, **evaluate.position_scores(z[:, 0], truth[:, 0], c.members.party),
                     "procrustes_r2_2d": evaluate.procrustes_r2(z, truth)})
    pd.DataFrame(rows).to_csv(RES / "synthetic_positions.csv", index=False)
    rows = []
    for name, (lab, info) in cluster.run_all(zl, c.votes).items():
        rows.append({"method": name, "k": info["k"], **evaluate.cluster_scores(lab, c.members.faction),
                     "party_ari": evaluate.cluster_scores(lab, c.members.party)["ari"]})
    from sklearn.cluster import KMeans
    lab = KMeans(6, n_init=10, random_state=0).fit_predict(cluster._std(zl))
    rows.append({"method": "kmeans (told k=6)", "k": 6, **evaluate.cluster_scores(lab, c.members.faction),
                 "party_ari": evaluate.cluster_scores(lab, c.members.party)["ari"]})
    pd.DataFrame(rows).to_csv(RES / "synthetic_clusters.csv", index=False)
    print(pd.read_csv(RES / "synthetic_positions.csv").round(3), pd.read_csv(RES / "synthetic_clusters.csv").round(3))


def heldout():
    m = matrix.build(118, "H")
    info = matrix.with_members(m, 118, "H")
    rows = [embed.heldout_accuracy(m, d, lam) for d in (1, 2, 3) for lam in (0.3, 1, 3, 10)]
    rows.append({"d": 0, "lam": None, "accuracy": embed.party_baseline_accuracy(m, info.party),
                 "log_loss": None, "note": "party-majority baseline"})
    pd.DataFrame(rows).to_csv(RES / "heldout_118H.csv", index=False)
    rows = []
    for c, ch in ((118, "H"), (118, "S"), (119, "H"), (119, "S")):
        m = matrix.build(c, ch)
        info = matrix.with_members(m, c, ch)
        right = (info.party == "Republican").to_numpy()
        for name, z in (("PCA", embed.pca(m, 2, right)), ("logistic PCA", embed.lpca(m, 2, right_mask=right)[0])):
            rows.append({"congress": c, "chamber": ch, "method": name,
                         **{f"dim1_{k}": v for k, v in evaluate.position_scores(z[:, 0], info.nominate_dim1, info.party).items()},
                         **{f"dim2_{k}": v for k, v in evaluate.position_scores(z[:, 1], info.nominate_dim2, info.party).items()}})
    pd.DataFrame(rows).to_csv(RES / "vs_nominate.csv", index=False)
    print(pd.read_csv(RES / "heldout_118H.csv").round(4), pd.read_csv(RES / "vs_nominate.csv").round(3).to_string())


def clusters():
    m = matrix.build(119, "H")
    info = matrix.with_members(m, 119, "H")

    def ef(v):
        return embed.lpca(v, 2, right_mask=(info.reindex(v.index).party == "Republican").to_numpy())[0]

    z = ef(m)
    lab, report = caucus.label(info)
    out = pd.DataFrame({"icpsr": info.index, "name": info.bioname, "party": info.party,
                        "state": info.state_abbrev, "district": info.district_code,
                        "caucus": lab.values, "dim1": z[:, 0], "dim2": z[:, 1],
                        "nominate_dim1": info.nominate_dim1, "nominate_dim2": info.nominate_dim2})
    rows = []
    for name, (labels, i) in cluster.run_all(z, m).items():
        out[name] = labels
        r = {"method": name, "k": i["k"], "party_ari": evaluate.cluster_scores(labels, info.party)["ari"]}
        for c, p in CAUCUS_TARGETS:
            r[f"{c} ({p[0]})"] = evaluate.caucus_match(labels, lab.values, info.party, c, p)["f1"]
        rows.append(r)
    gm = cluster.gmm(z)
    out["gmm_max_membership"] = gm[1]["membership"].max(1)
    pd.Series(gm[1]["bic"]).to_csv(RES / "gmm_bic_119H.csv", header=["bic"], index_label="k")
    stab = cluster.bootstrap_stability(m, ef, n_boot=20)
    t = pd.DataFrame(rows)
    t["stability_median_ari"] = t.method.map(lambda k: float(np.median(stab[k])))
    t["stability_p10_ari"] = t.method.map(lambda k: float(np.percentile(stab[k], 10)))
    t.to_csv(RES / "clusters_119H.csv", index=False)
    out.to_csv(RES / "members_119H.csv", index=False)
    (RES / "caucus_match_119H.json").write_text(json.dumps(report, indent=2) + "\n")
    print(t.round(2).to_string())


def polarization_series():
    s = polarization.series()
    s.to_csv(RES / "polarization.csv", index=False)
    print(s.round(3).to_string())


STEPS = {"synthetic": synthetic, "heldout": heldout, "clusters": clusters, "polarization": polarization_series}

if __name__ == "__main__":
    for step in sys.argv[1:] or STEPS:
        STEPS[step]()
