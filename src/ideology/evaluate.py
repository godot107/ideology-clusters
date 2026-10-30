"""Scoring positions and clusters against a known answer."""
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def position_scores(est, truth, party):
    """est, truth: 1-D arrays (dim 1). |Pearson| overall, and Spearman *within each party*:
    the real test, since any method separates the parties."""
    est, truth, party = np.asarray(est), np.asarray(truth), np.asarray(party)
    ok = ~np.isnan(est) & ~np.isnan(truth)
    out = {"overall_r": abs(float(np.corrcoef(est[ok], truth[ok])[0, 1]))}
    for p in pd.unique(party[ok]):
        sel = ok & (party == p)
        if sel.sum() >= 10:
            out[f"within_{p}_spearman"] = abs(float(spearmanr(est[sel], truth[sel])[0]))
    return out


def procrustes_r2(est, truth):
    """How well a 2-D estimate matches 2-D truth after the best rotation/reflection and scaling
    (both sides standardised). 1.0 = same configuration."""
    A = (est - est.mean(0)) / est.std(0)
    B = (truth - truth.mean(0)) / truth.std(0)
    R, _ = orthogonal_procrustes(A, B)
    resid = ((A @ R - B) ** 2).sum()
    return float(1 - resid / (B ** 2).sum())


def cluster_scores(labels, reference):
    """Agreement between two labelings that don't share label names. Noise (-1) counts as its own
    group. ARI: 0 = chance, 1 = identical. NMI: shared information, 0-1."""
    return {"ari": float(adjusted_rand_score(reference, labels)),
            "nmi": float(normalized_mutual_info_score(reference, labels))}


def caucus_match(labels, caucus_labels, party, caucus, caucus_party):
    """How well one cluster captures one caucus, within the caucus's party. For each cluster:
    recall = share of the caucus in it, precision = share of the cluster (party members only) that's
    in the caucus; report the cluster with the best F1. A caucus scattered across clusters, or
    swallowed by a huge one, scores low."""
    labels, party = np.asarray(labels), np.asarray(party)
    inc = np.asarray([caucus in c for c in caucus_labels])
    sel = party == caucus_party
    best = {"f1": 0.0}
    for c in np.unique(labels[sel]):
        if c == -1:
            continue
        in_c = sel & (labels == c)
        tp = (in_c & inc).sum()
        if tp == 0:
            continue
        p, r = tp / in_c.sum(), tp / (sel & inc).sum()
        f1 = 2 * p * r / (p + r)
        if f1 > best["f1"]:
            best = {"f1": float(f1), "precision": float(p), "recall": float(r), "cluster": int(c),
                    "cluster_size": int(in_c.sum())}
    return best


def breakaway(votes, positions, party, min_dissent=0.05, min_voters=30):
    """Who breaks with their own party, and which dimension says so. For each party and roll call with
    at least `min_dissent` of the party voting against the party majority: the correlation between
    "voted with my party's majority" (1/0) and each position column. Positive = members high on that
    dimension stay with the party; negative = they break away. Model-free: uses only the raw votes and
    the positions."""
    party = pd.Series(np.asarray(party), index=votes.index)
    rows = []
    for p in pd.unique(party):
        sub = votes[party == p]
        for col in sub.columns:
            y = sub[col].dropna()
            if len(y) < min_voters:
                continue
            with_party = (y == float(y.mean() >= 0.5)).astype(float)
            dissent = 1 - with_party.mean()
            if dissent < min_dissent:
                continue
            row = {"party": p, "rollnumber": col, "voters": len(y), "dissent": float(dissent),
                   "party_yea": float(y.mean())}
            for d in positions.columns:
                row[f"r_{d}"] = float(np.corrcoef(with_party, positions.loc[y.index, d])[0, 1])
            rows.append(row)
    return pd.DataFrame(rows)
