"""Placing members in a low-dimensional "ideology space" from their votes.

- pca:   PCA on the vote matrix with each missed vote filled by that roll call's average.
         Treats yes/no as numbers and drags frequent absentees toward the middle.
- lpca:  logistic PCA (binary matrix factorisation). Models P(yea) = sigmoid(a_j + x_i . b_j) and
         simply leaves missed votes out of the fit, so absentees aren't pulled anywhere.
Both return a members x d array whose first column is oriented so Republicans are positive
(sign is arbitrary in both methods; orientation is a presentation choice, not information).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import PCA


def orient(z, right_mask):
    """Flip each column so the right_mask group (Republicans) has the higher mean on dim 1, and the
    smaller group's... only dim 1 is oriented; other dims keep their arbitrary sign."""
    z = z.copy()
    if right_mask is not None and z[right_mask, 0].mean() < z[~right_mask, 0].mean():
        z[:, 0] *= -1
    return z


def pca(votes, d=2, right_mask=None):
    X = votes.fillna(votes.mean()).to_numpy()
    X = X - X.mean(axis=0)
    return orient(PCA(d).fit_transform(X), right_mask)


def _fit_raw(votes, d, lam, seed=0, maxiter=3000):
    """Fit by L-BFGS on the masked Bernoulli log-likelihood plus an L2 penalty `lam` on member and
    roll-call vectors (fixes the scale and keeps unanimous-looking roll calls finite)."""
    Y = votes.to_numpy()
    M = ~np.isnan(Y)
    Y0 = np.where(M, Y, 0.0)
    n, m = Y.shape
    rng = np.random.default_rng(seed)
    # start from PCA so the optimiser begins near a sensible answer
    Z0 = pca(votes, d)
    Z0 = Z0 / Z0.std(axis=0)
    params0 = np.concatenate([Z0.ravel(), rng.normal(0, 0.1, m * d), np.zeros(m)])

    def unpack(p):
        U = p[: n * d].reshape(n, d)
        V = p[n * d: n * d + m * d].reshape(m, d)
        a = p[n * d + m * d:]
        return U, V, a

    def f(p):
        U, V, a = unpack(p)
        T = U @ V.T + a
        # log(1 + exp(T)) - y*T, computed stably; masked
        ll = np.where(M, np.logaddexp(0, T) - Y0 * T, 0.0)
        G = np.where(M, 1 / (1 + np.exp(-T)) - Y0, 0.0)
        loss = ll.sum() + lam * ((U ** 2).sum() + (V ** 2).sum())
        gU = G @ V + 2 * lam * U
        gV = G.T @ U + 2 * lam * V
        ga = G.sum(axis=0)
        return loss, np.concatenate([gU.ravel(), gV.ravel(), ga])

    res = minimize(f, params0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    U, V, a = unpack(res.x)
    return U, V, a, res


def lpca(votes, d=2, lam=1.0, right_mask=None, seed=0, maxiter=3000):
    U, V, a, res = _fit_raw(votes, d, lam, seed, maxiter)
    # rotate to principal axes of the member positions, so dim 1 is the main dimension
    Uc = U - U.mean(axis=0)
    _, _, Wt = np.linalg.svd(Uc, full_matrices=False)
    return orient(Uc @ Wt.T, right_mask), {"converged": bool(res.success), "loss": float(res.fun),
                                           "iterations": int(res.nit)}


def heldout_accuracy(votes, d, lam, frac=0.1, seed=0):
    """Hide a random 10% of the recorded votes, fit on the rest, and score how well the model predicts
    the hidden ones (accuracy and log loss). Picks d and lam on real data, where there's no answer key."""
    rng = np.random.default_rng(seed)
    Y = votes.to_numpy()
    obs = np.argwhere(~np.isnan(Y))
    hide = obs[rng.random(len(obs)) < frac]
    Yt = Y.copy()
    Yt[hide[:, 0], hide[:, 1]] = np.nan
    train = votes.__class__(Yt, index=votes.index, columns=votes.columns)
    U, V, a, _ = _fit_raw(train, d, lam)
    T = (U @ V.T + a)[hide[:, 0], hide[:, 1]]
    p = 1 / (1 + np.exp(-T))
    y = Y[hide[:, 0], hide[:, 1]]
    eps = 1e-9
    return {"d": d, "lam": lam, "accuracy": float(((p > 0.5) == (y == 1)).mean()),
            "log_loss": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))}


def party_baseline_accuracy(votes, party, frac=0.1, seed=0):
    """The simplest rival: predict each hidden vote as the member's party majority on that roll call
    (from the visible votes). Same hidden votes as heldout_accuracy (same seed and frac)."""
    rng = np.random.default_rng(seed)
    Y = votes.to_numpy()
    obs = np.argwhere(~np.isnan(Y))
    hide = obs[rng.random(len(obs)) < frac]
    Yt = Y.copy()
    Yt[hide[:, 0], hide[:, 1]] = np.nan
    party = np.asarray(party)
    pred = np.full(Y.shape, np.nan)
    for p in np.unique(party):
        rows = party == p
        with np.errstate(invalid="ignore"):
            pred[rows] = np.nanmean(Yt[rows], axis=0) >= 0.5
    y = Y[hide[:, 0], hide[:, 1]]
    return float((pred[hide[:, 0], hide[:, 1]] == y).mean())
