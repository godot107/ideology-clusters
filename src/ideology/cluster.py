"""Clustering members, five ways. Each method chooses its own number of groups by its standard rule,
so none is handed the answer:

- kmeans:   round clusters of similar size; k by best silhouette score (2-8).
- gmm:      Gaussian mixture (elliptical clusters, soft membership); k by lowest BIC (1-8).
- hdbscan:  density-based; finds tight groups and labels members between them -1 ("noise").
- ward:     hierarchical (agglomerative, Ward linkage); cut at the k with the best silhouette.
- network:  no embedding: Louvain communities on a co-voting graph linking each member to the 15
            colleagues they agree with most often.

Embedding-based methods run on standardised positions (e.g. logistic PCA, 2-D).
"""
import networkx as nx
import numpy as np
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

K_RANGE = range(2, 9)


def _std(z):
    return (z - z.mean(0)) / z.std(0)


def kmeans(z, seed=0):
    z = _std(z)
    fits = {k: KMeans(k, n_init=10, random_state=seed).fit(z) for k in K_RANGE}
    k = max(fits, key=lambda k: silhouette_score(z, fits[k].labels_))
    return fits[k].labels_, {"k": k}


def gmm(z, seed=0):
    z = _std(z)
    fits = {k: GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(z)
            for k in range(1, 9)}
    k = min(fits, key=lambda k: fits[k].bic(z))
    g = fits[k]
    return g.predict(z), {"k": k, "membership": g.predict_proba(z), "bic": {k: f.bic(z) for k, f in fits.items()}}


def hdbscan(z, min_cluster_size=10):
    lab = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(_std(z))
    return lab, {"k": int(len(set(lab)) - (-1 in lab)), "noise": int((lab == -1).sum())}


def ward(z):
    z = _std(z)
    fits = {k: AgglomerativeClustering(k, linkage="ward").fit(z).labels_ for k in K_RANGE}
    k = max(fits, key=lambda k: silhouette_score(z, fits[k]))
    return fits[k], {"k": k}


def agreement(votes):
    """Share of roll calls two members both voted on where they voted the same way."""
    Y = votes.to_numpy()
    M = (~np.isnan(Y)).astype(float)
    Yy = np.nan_to_num(Y) * M
    Yn = (1 - np.nan_to_num(Y)) * M
    both = M @ M.T
    same = Yy @ Yy.T + Yn @ Yn.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(both > 0, same / both, np.nan)


def network(votes, k_neighbors=15, seed=0):
    A = agreement(votes)
    np.fill_diagonal(A, -np.inf)
    G = nx.Graph()
    G.add_nodes_from(range(len(A)))
    for i in range(len(A)):
        for j in np.argsort(-np.nan_to_num(A[i], nan=-1))[:k_neighbors]:
            G.add_edge(i, int(j), weight=float(A[i, j]))
    comms = nx.community.louvain_communities(G, weight="weight", seed=seed)
    lab = np.empty(len(A), int)
    for c, members in enumerate(sorted(comms, key=len, reverse=True)):
        lab[list(members)] = c
    return lab, {"k": len(comms), "modularity": nx.community.modularity(G, comms, weight="weight")}


def run_all(z, votes, seed=0):
    return {"kmeans": kmeans(z, seed), "gmm": gmm(z, seed), "hdbscan": hdbscan(z),
            "ward": ward(z), "network": network(votes, seed=seed)}


def bootstrap_stability(votes, embed_fn, n_boot=20, seed=0):
    """Resample roll calls with replacement, re-embed, re-cluster with every method; ARI of each
    bootstrap's labels against the full-data labels. Low = the 'factions' depend on which votes
    happened to be held. Returns {method: array of ARIs}."""
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    base = {k: v[0] for k, v in run_all(embed_fn(votes), votes, seed).items()}
    out = {k: [] for k in base}
    for b in range(n_boot):
        cols = rng.choice(votes.shape[1], votes.shape[1], replace=True)
        vb = votes.iloc[:, cols].copy()
        vb.columns = range(vb.shape[1])
        for k, (lab, _) in run_all(embed_fn(vb), vb, seed).items():
            out[k].append(adjusted_rand_score(base[k], lab))
    return {k: np.array(v) for k, v in out.items()}
