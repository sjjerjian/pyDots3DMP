#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 09:56:40 2023

@author: stevenjerjian

"""


import numpy as np
from   scipy.stats import (invwishart,
                           multivariate_normal)
from   time import perf_counter


def multiple_logpdfs(x, means, covs):
    """Compute multivariate normal log PDF over multiple sets of parameters.
    """
    # NumPy broadcasts `eigh`.
    vals, vecs = np.linalg.eigh(covs)

    # Compute the log determinants across the second axis.
    logdets    = np.sum(np.log(vals), axis=1)

    # Invert the eigenvalues.
    valsinvs   = 1./vals

    # Add a dimension to `valsinvs` so that NumPy broadcasts appropriately.
    Us         = vecs * np.sqrt(valsinvs)[:, None]
    devs       = x - means

    # Use `einsum` for matrix-vector multiplications across the first dimension.
    devUs      = np.einsum('ni,nij->nj', devs, Us)

    # Compute the Mahalanobis distance by squaring each term and summing.
    mahas      = np.sum(np.square(devUs), axis=1)

    # Compute and broadcast scalar normalizers.
    dim        = len(vals[0])
    log2pi     = np.log(2 * np.pi)
    return -0.5 * (dim * log2pi + mahas + logdets)




dim   = 3
n     = 100
# Generate random data, means, and positive-definite covariance matrices.
x     = np.random.normal(size=dim)
means = np.random.random(size=(n, dim))
covs  = invwishart(df=dim, scale=np.eye(dim)).rvs(size=n)
ps1   = np.empty(n)

# Compute and time probabilities the slow way.
s = perf_counter()
for i, (m, c) in enumerate(zip(means, covs)):
    ps1[i] = multivariate_normal(m, c).logpdf(x)
t1 = perf_counter() - s

# Compute and time probabilities the fast way.
s = perf_counter()
ps2 = multiple_logpdfs(x, means, covs)
t2 = perf_counter() - s

print(t1 / t2)
assert(np.allclose(ps1, ps2))
