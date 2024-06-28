#!/usr/bin/env python3
"""RMSProp RMSProp """

import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm

    Args:
        alpha (float): the learning rate
        beta2 (float): the RMSProp weight
        epsilon (float): small number to avoid division by zero
        var (numpy.ndarray): the variable to be updated
        grad (numpy.ndarray): the gradient of var
        s (numpy.ndarray): the previous second moment of var

    Returns:
        tuple: updated variable and updated second moment
    """
    s = beta2 * s + (1 - beta2) * grad ** 2
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
