#!/usr/bin/env python3
""" Module that creates a training operation in TensorFlow. """
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    Creates a training operation for a given loss and learning rate.

    Args:
        loss: The loss value to minimize.
        alpha: The learning rate for the optimizer.

    Returns:
        The training operation that minimizes the loss.
    """
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    training = optimizer.minimize(loss)
    return training
