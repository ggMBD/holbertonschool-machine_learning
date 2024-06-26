#!/usr/bin/env python3
""" 8. NeuralNetwork """
import numpy as np


class NeuralNetwork:
    """Class NeuralNetwork"""

    def __init__(self, nx, nodes):
        """Constructor method"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ Returns the values of the weights vector """
        return self.__W1

    @property
    def b1(self):
        """ Returns the value of the bias neuron """
        return self.__b1

    @property
    def A1(self):
        """ Returns the value of the activation function output """
        return self.__A1

    @property
    def W2(self):
        """ Returns the values of the weights vector """
        return self.__W2

    @property
    def b2(self):
        """ Returns the value of the bias neuron """
        return self.__b2

    @property
    def A2(self):
        """ Returns the value of the activation function output """
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        cost = -np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A))) / m
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        _, A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = Y.shape[1]
        dz2 = A2 - Y
        dw2 = np.matmul(A1, dz2.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m
        dz1 = np.matmul(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = np.matmul(X, dz1.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m
        self.__W2 -= alpha * dw2.T
        self.__b2 -= alpha * db2
        self.__W1 -= alpha * dw1.T
        self.__b1 -= alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the neuron """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 1 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x = np.arange(0, iterations + 1, step)
            y = np.empty((iterations // step) + 1, dtype=float)

        for i in range(iterations):
            _, A = self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print("Cost after {} iterations: {}".format(i, cost))
            if graph and i % step == 0:
                y[i // step] = cost
        
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after {} iterations: {}".format(iterations, cost))
        if graph:
            y[-1] = cost
            plt.plot(x, y, 'b-')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

