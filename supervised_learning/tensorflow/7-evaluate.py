#!/usr/bin/env python3
""" Module that evaluates the output of a neural network. """
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    Evaluate the accuracy and loss of a trained model using
        the provided input data.

    Args:
        X (numpy.ndarray): The input data.
        Y (numpy.ndarray): The target data.
        save_path (str): The path to the saved model.

    Returns:
        tuple: A tuple containing the predicted values, accuracy, and loss.

    Raises:
        None

    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        pred = sess.run(y_pred, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        cost = sess.run(loss, feed_dict={x: X, y: Y})
    return pred, acc, cost
