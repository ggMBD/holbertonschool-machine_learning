#!/usr/bin/env python3
"""Normalization Constants"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix.

    Args:
        X (numpy.ndarray): The input matrix.

    Returns:
        tuple: A tuple containing the mean and
        standard deviation of each column in the matrix.
    """
    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)
    return mean, std_dev
