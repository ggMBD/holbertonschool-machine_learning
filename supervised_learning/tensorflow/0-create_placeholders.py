#!/usr/bin/env python3
""" Module that creates placeholders for
    input features and labels in TensorFlow."""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates placeholders for input features (x) and labels (y) in TensorFlow.

    Args:
        nx (int): The number of input features.
        classes (int): The number of classes.

    Returns:
        x (tf.Tensor): The input placeholder tensor of shape [None, nx].
        y (tf.Tensor): The label placeholder tensor of shape [None, classes].
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y
