#!/usr/bin/env python3
""" 0. Likelihood """
import numpy as np


def likelihood(x, n, P):
    """ calculates the likelihood of obtaining this data given various
    hypothetical probabilities of developing severe side effects"""

    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError('All values in P must be in the range [0, 1]')

    likelihood = np.zeros(P.shape)

    for i, p in enumerate(P):
        likelihood[i] = np.math.factorial(
            n) / (np.math.factorial(x) * np.math.factorial(n - x)) \
            * (p ** x) * ((1 - p) ** (n - x))

    return likelihood
