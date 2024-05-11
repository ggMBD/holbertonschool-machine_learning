#!/usr/bin/env python3
""" Module that creates a dense layer in TensorFlow. """
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Creates a dense layer to give the output
    of our neural network.

    Args:
        prev (Tensor): The input tensor for the layer.
        n (int): The number of units/neurons in the layer.
        activation (str): The activation function to be used.

    Returns:
        Tensor: The output tensor of the layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=init, name='layer')
    return layer(prev)
