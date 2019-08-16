"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    mu = mixture.mu
    var = mixture.var
    p = mixture.p

    n, d = X.shape
    k, = p.shape

    not_normalized = np.array([[p[my_k]*(1/(2*np.pi*var[my_k]))**(d/2)*np.exp(-1/(2*var[my_k])*np.linalg.norm(mu[my_k]-my_x)**2) for my_k in range(k)] for my_x in X])
    norm = not_normalized.sum(axis=1)
    post = (not_normalized.transpose()/norm).transpose()
    ll = (post * np.log(not_normalized/post)).sum()

    return post, ll


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n_hat = post.sum(axis=0)
    _, K = post.shape
    n, d = X.shape
    p = n_hat/n

    f_out = np.array([[j*X[i] for j in post[i]] for i in range(n)]).sum(axis=0)
    mu = (f_out.transpose()/n_hat).transpose()

    s_out = np.array([[post[i, j]*np.linalg.norm(X[i]-mu[j])**2 for j in range(K)] for i in range(n)]).sum(axis=0)
    var = s_out/(d*n_hat)
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



