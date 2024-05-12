#!/usr/bin/env python3
""" Module that calculates the accuracy of a prediction. """
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y (tf.Tensor): The true labels.
        y_pred (tf.Tensor): The predicted labels.

    Returns:
        tf.Tensor: The accuracy of the prediction.
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
