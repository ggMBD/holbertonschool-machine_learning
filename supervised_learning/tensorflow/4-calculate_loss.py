#!/usr/bin/env python3
""" Module that calculates the loss of a prediction."""
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    Calculates the loss with respect to the final predicted labels of that step.

    Args:
        y: The true labels, a tensor of shape (batch_size, num_classes).
        y_pred: The predicted labels, a tensor of shape (batch_size, num_classes).

    Returns:
        The calculated loss, a scalar tensor.

    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (labels=y, logits=y_pred))
    return loss
