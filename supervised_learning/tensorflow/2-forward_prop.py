#!/usr/bin/env python3
""" Module that performs forward propagation in a neural network. """
import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Performs forward propagation in a neural network.

    Args:
        x (tf.Tensor): The input tensor.
        layer_sizes (list):
            A list of integers representing the sizes of each layer.
        activations (list):
            A list of activation functions for each layer.

    Returns:
        tf.Tensor: The output tensor after forward propagation.
    """
    for i in range(len(layer_sizes)):
        output = create_layer(x, layer_sizes[i], activations[i])
        x = output
    return output
