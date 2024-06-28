#!/usr/bin/env python3
"""RMSProp Upgrade."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Sets up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): The learning rate.
        beta2 (float): The decay rate for the moving average of
                        squared gradients.
        epsilon (float): A small constant for numerical stability.

    Returns:
        optimizer: The RMSProp optimizer.

    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
    return optimizer
