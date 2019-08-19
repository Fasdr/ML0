"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    mu = mixture.mu
    var = mixture.var
    p = mixture.p

    k, = var.shape
    n, d = X.shape

    mask = np.where(X != 0, 1, 0)
    dim_X = mask.sum(axis=1)
    dim_X = dim_X[:, None]
    dim_X = np.tile(dim_X, (1, k))

    new_X = X[:, :, None]
    new_X = np.tile(new_X, (1, 1, k))
    new_X = np.swapaxes(new_X, 1, 2)

    new_var = var[:, None]
    new_var = np.tile(new_var, (1, n))
    new_var = np.swapaxes(new_var, 0, 1)

    new_mu = mu[:, :, None]
    new_mu = np.tile(new_mu, (1, 1, n))
    new_mu = np.swapaxes(new_mu, 0, 2)
    new_mu = np.swapaxes(new_mu, 1, 2)

    args = np.where(new_X == 0, 0, new_X - new_mu)
    norms = (args ** 2).sum(axis=2)
    N = ((2 * np.pi * new_var) ** (-dim_X / 2)) * np.exp(-((2 * new_var) ** (-1)) * norms)

    new_p = p[:, None]
    new_p = np.tile(new_p, (1, n))
    new_p = np.swapaxes(new_p, 0, 1)
    f = np.log((1e-16 + new_p) * N)
    loss = (np.exp(f).sum(axis=1))
    out_loss = np.log(loss).sum()
    new_f = np.log(np.tile(loss[:, None], (1, k)))
    final_f = f - new_f
    out_p = np.exp(final_f)

    return out_p, out_loss


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    old_mu = mixture.mu

    _, k = post.shape
    n, d = X.shape

    p = post.sum(axis=0) / n
    new_X = np.tile(X[:, :, None], (1, 1, k))
    mask = np.where(new_X != 0, 1, 0)
    new_post = np.swapaxes(np.tile(post[:, :, None], (1, 1, d)), 1, 2)

    mu = np.swapaxes(((new_post * new_X).sum(axis=0) / (new_post * mask).sum(axis=0)), 0, 1)

    checker = np.swapaxes((mask*new_post).sum(axis=0), 0, 1)

    mu = np.where(checker >= 1, mu, old_mu)

    new_mu = np.swapaxes(np.tile(mu[:, :, None], (1, 1, n)), 0, 2)

    norms = ((new_X - mask * new_mu) ** 2).sum(axis=1)
    w_norms = (norms * post).sum(axis=0)
    new_mask = mask.sum(axis=1)

    var = (((new_mask * post).sum(axis=0)) ** (-1)) * w_norms
    var = np.where(var > min_variance, var, min_variance)
    return GaussianMixture(mu, var, p)

def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_log = float('-inf')
    post, new_log = estep(X, mixture)
    while new_log - old_log > 10**(-6)*abs(new_log):
        mixture = mstep(X, post)
        old_log = new_log
        post, new_log = estep(X, mixture)

    return mixture, post, new_log

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    raise NotImplementedError
