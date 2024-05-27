#!/usr/bin/env python3
"""L2 Regularization Cost"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """ calculates the cost of a neural
    network with L2 regularization"""
    reg_loss = []
    for layer in model.layers:
        reg_loss.append(tf.reduce_sum(layer.losses) + cost)
    return tf.stack(reg_loss[1:])
