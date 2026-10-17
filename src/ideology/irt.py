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

numpyro.set_host_device_count(1)


def model(i, j, y, n, m):
    x = numpyro.sample("x", dist.Normal(0, 1).expand([n]))
    alpha = numpyro.sample("alpha", dist.Normal(0, 5).expand([m]))
    beta = numpyro.sample("beta", dist.Normal(0, 5).expand([m]))
    numpyro.sample("y", dist.Bernoulli(logits=beta[j] * x[i] - alpha[j]), obs=y)


def long_form(votes):
    Y = votes.to_numpy()
    ii, jj = np.nonzero(~np.isnan(Y))
    return ii, jj, Y[ii, jj].astype(np.int32), Y.shape


def fit(votes, right_mask, warmup=500, samples=500, seed=0, init_x=None):
    """Returns posterior draws of x (samples x members), oriented, plus diagnostics."""
    ii, jj, y, (n, m) = long_form(votes)
    init = None
    if init_x is not None:   # start chains at a good answer (e.g. logistic PCA dim 1, standardised)
        x0 = (init_x - init_x.mean()) / init_x.std()
        init = {"x": jnp.asarray(x0), "alpha": jnp.zeros(m), "beta": jnp.zeros(m)}
    kernel = NUTS(model, init_strategy=numpyro.infer.init_to_value(values=init) if init else
                  numpyro.infer.init_to_median())
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), jnp.asarray(ii), jnp.asarray(jj), jnp.asarray(y), n, m)
    x = np.array(mcmc.get_samples()["x"])  # writable copy
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
