"""Bayesian ideal points (Clinton, Jackman & Rivers 2004) in NumPyro: one position per member,
with a credible interval.

    x_i      ~ Normal(0, 1)             member i's position (the prior fixes the scale)
    alpha_j  ~ Normal(0, 5)             roll call j's cut point
    beta_j   ~ Normal(0, 5)             roll call j's direction and sharpness
    P(yea_ij) = sigmoid(beta_j * x_i - alpha_j)

Missed votes are left out of the likelihood. The model can't tell left from right (flipping every
x and beta gives the same fit), so samples are oriented afterwards: Republicans' mean positive.
Inference: NUTS (exact up to Monte Carlo error) with a mean-field SVI fallback for speed checks.
"""
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

numpyro.set_host_device_count(2)   # lets 2 chains run in parallel on CPU


def model(i, j, y, n, m):
    x = numpyro.sample("x", dist.Normal(0, 1).expand([n]))
    alpha = numpyro.sample("alpha", dist.Normal(0, 5).expand([m]))
    beta = numpyro.sample("beta", dist.Normal(0, 5).expand([m]))
    numpyro.sample("y", dist.Bernoulli(logits=beta[j] * x[i] - alpha[j]), obs=y)


def long_form(votes):
    Y = votes.to_numpy()
    ii, jj = np.nonzero(~np.isnan(Y))
    return ii, jj, Y[ii, jj].astype(np.int32), Y.shape


def fit(votes, right_mask, warmup=500, samples=500, seed=0, init_x=None, chains=1):
    """Returns posterior draws of x (samples x members), oriented, plus diagnostics."""
    ii, jj, y, (n, m) = long_form(votes)
    init = None
    if init_x is not None:   # start chains at a good answer (e.g. logistic PCA dim 1, standardised)
        x0 = (init_x - init_x.mean()) / init_x.std()
        init = {"x": jnp.asarray(x0), "alpha": jnp.zeros(m), "beta": jnp.zeros(m)}
    kernel = NUTS(model, init_strategy=numpyro.infer.init_to_value(values=init) if init else
                  numpyro.infer.init_to_median())
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel" if chains > 1 else "sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(ii), jnp.asarray(jj), jnp.asarray(y), n, m)
    x = np.array(mcmc.get_samples()["x"])  # writable copy; chains concatenated
    # orient each draw (a chain can't flip mid-run, but be safe)
    flip = x[:, right_mask].mean(1) < x[:, ~right_mask].mean(1)
    x[flip] *= -1
    return x, diagnostics(np.asarray(mcmc.get_samples(group_by_chain=True)["x"]),
                          int(mcmc.get_extra_fields()["diverging"].sum()))


def diagnostics(chains, divergences):
    """chains: (n_chains, draws, members). Split R-hat works with one chain (it compares the two
    halves); near 1.0 = the sampler settled. ESS = how many independent draws the samples are worth."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    try:
        return {"max_split_rhat": float(np.nanmax(split_gelman_rubin(chains))),
                "min_ess": float(np.nanmin(effective_sample_size(chains))),
                "divergences": divergences}
    except Exception as e:  # never lose a finished run to a diagnostics error
        return {"diagnostics_error": repr(e), "divergences": divergences}


def summarize(x):
    """Posterior mean and 90% credible interval per member."""
    return {"mean": x.mean(0), "lo": np.percentile(x, 5, axis=0), "hi": np.percentile(x, 95, axis=0)}


def model2d(i, j, y, n, m, anchor):
    """Two-dimensional version: x_i and beta_j are 2-vectors. The fit is unchanged by rotating every
    x and beta together, so one roll call (`anchor`, a party-line vote) is held to dimension 1
    (its beta_2 = 0). That fixes the rotation; reflections are fixed after sampling."""
    x = numpyro.sample("x", dist.Normal(0, 1).expand([n, 2]).to_event(2))
    alpha = numpyro.sample("alpha", dist.Normal(0, 5).expand([m]))
    beta = numpyro.sample("beta", dist.Normal(0, 5).expand([m, 2]).to_event(2))
    beta = beta.at[anchor, 1].set(0.0)
    numpyro.sample("y", dist.Bernoulli(logits=(beta[j] * x[i]).sum(-1) - alpha[j]), obs=y)


def anchor_rollcall(votes, party):
    """The roll call that best splits the parties (largest gap in yea share): a dimension-1 vote."""
    party = np.asarray(party)
    gap = np.abs(np.nanmean(votes.to_numpy()[party == "R"], 0) - np.nanmean(votes.to_numpy()[party == "D"], 0))
    return int(np.nanargmax(gap))


def fit2d(votes, party, init_xy, warmup=500, samples=500, seed=0, chains=1):
    """Posterior draws of 2-D positions (draws x members x 2). Dim 1 oriented Republicans positive;
    dim 2 oriented to agree in sign with init_xy's dim 2 (a presentation choice)."""
    ii, jj, y, (n, m) = long_form(votes)
    party = np.asarray(party)
    right = (party == "R") | (party == "Republican")
    anchor = anchor_rollcall(votes, np.where(right, "R", "D"))
    x0 = (init_xy - init_xy.mean(0)) / init_xy.std(0)
    init = {"x": jnp.asarray(x0), "alpha": jnp.zeros(m), "beta": jnp.zeros((m, 2))}
    kernel = NUTS(model2d, init_strategy=numpyro.infer.init_to_value(values=init))
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel" if chains > 1 else "sequential", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(ii), jnp.asarray(jj), jnp.asarray(y), n, m, anchor)
    x = np.array(mcmc.get_samples()["x"])
    for d in range(x.shape[0]):
        if x[d, right, 0].mean() < x[d, ~right, 0].mean():
            x[d, :, 0] *= -1
        if np.corrcoef(x[d, :, 1], x0[:, 1])[0, 1] < 0:
            x[d, :, 1] *= -1
    chains_x = np.asarray(mcmc.get_samples(group_by_chain=True)["x"])
    diag = diagnostics(chains_x.reshape(chains_x.shape[0], chains_x.shape[1], -1),
                       int(mcmc.get_extra_fields()["diverging"].sum()))
    diag["anchor_rollcall_column"] = anchor
    return x, diag


def coverage(draws, truth, groups):
    """Share of true positions inside the 90% interval, per group, after one linear map from the
    posterior mean to the truth (the model's scale and origin are arbitrary). draws: (draws, members)."""
    s = summarize(draws)
    a, b = np.polyfit(truth, s["mean"], 1)
    t = a * np.asarray(truth) + b
    inside = (t >= s["lo"]) & (t <= s["hi"])
    import pandas as pd
    return pd.Series(inside, index=np.asarray(groups)).groupby(level=0).mean()


def coverage2d(draws, truth, groups):
    """2-D version of `coverage`: one affine map (2x2 + shift, least squares) from the posterior mean
    to the true positions, applied to every draw, so a small rotation between the model's axes and the
    truth's doesn't count as a miss. Returns per-group coverage on each dimension. draws: (draws, n, 2)."""
    import pandas as pd
    mean = draws.mean(0)
    A = np.c_[mean, np.ones(len(mean))]
    W, *_ = np.linalg.lstsq(A, np.asarray(truth), rcond=None)
    mapped = np.einsum("dnk,kj->dnj", np.concatenate([draws, np.ones(draws.shape[:2] + (1,))], -1), W)
    lo, hi = np.percentile(mapped, 5, axis=0), np.percentile(mapped, 95, axis=0)
    inside = (np.asarray(truth) >= lo) & (np.asarray(truth) <= hi)
    return pd.DataFrame(inside, index=np.asarray(groups), columns=["dim 1", "dim 2"]).groupby(level=0).mean()


def positions_table(draws, votes, info):
    """One row per member: posterior mean, 90% range, votes cast, DW-NOMINATE dim 1 for comparison."""
    s = summarize(draws)
    return info[["bioname", "party", "state_abbrev"]].assign(
        position=s["mean"], lo=s["lo"], hi=s["hi"], width=s["hi"] - s["lo"],
        votes_cast=votes.notna().sum(1).values, nominate_dim1=info.nominate_dim1)


def standardize(draws):
    """Rescale every draw to mean 0, sd 1 across members (per dimension). The model pins the axis's
    location and scale only weakly through the prior, so chains can drift by a shift and stretch that
    changes nobody's relative position; this is the usual identification (Clinton, Jackman & Rivers
    2004). draws: (draws, members) or (draws, members, dims)."""
    return (draws - draws.mean(1, keepdims=True)) / draws.std(1, keepdims=True)


def chain_diagnostics(draws, chains):
    """Split R-hat and ESS computed on (already oriented/standardised) draws, chains concatenated."""
    from numpyro.diagnostics import effective_sample_size, split_gelman_rubin
    ch = draws.reshape(chains, -1, int(np.prod(draws.shape[1:])))
    return {"max_split_rhat": float(np.nanmax(split_gelman_rubin(ch))),
            "min_ess": float(np.nanmin(effective_sample_size(ch)))}


def align(draws, reference):
    """Rotate/reflect every 2-D draw onto `reference` (members x 2) by orthogonal Procrustes, after
    centring and scaling both. Positions are only defined up to rotation; a single anchor pins it weakly,
    so chains drift slowly around it. Alignment removes that drift without changing anyone's position
    relative to anyone else. The reference must not come from the answer key (use logistic PCA)."""
    from scipy.linalg import orthogonal_procrustes
    ref = (reference - reference.mean(0)) / reference.std(0)
    out = np.empty_like(draws)
    for d in range(len(draws)):
        z = draws[d] - draws[d].mean(0)
        z = z / np.sqrt((z ** 2).sum() / len(z) / 2)       # overall scale, keeps the dims' ratio
        R, _ = orthogonal_procrustes(z, ref)
        out[d] = z @ R
    return out
