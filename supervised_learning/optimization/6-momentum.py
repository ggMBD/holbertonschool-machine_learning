#!/usr/bin/env python3
"""Momentum Upgraded"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Sets up the gradient descent with momentum
    optimization algorithm in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta1 (float): The momentum parameter.

    Returns:
        optimizer: The optimizer object for gradient descent with momentum.
    """
    optimizer = tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
    return optimizer
