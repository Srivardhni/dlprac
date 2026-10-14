"""Experiment 1: Perceptron for AND gate."""
import numpy as np
import matplotlib.pyplot as plt

def run():
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([0,0,0,1])
    weights = np.zeros(2)
    bias = 0.0
    learning_rate = 0.1
    for epoch in range(10):
        for xi, target in zip(X, y):
            pred = int(np.dot(xi, weights) + bias >= 1)
            error = target - pred
            weights += learning_rate * error * xi
            bias += learning_rate * error
    print("Weights:", weights)
    print("Bias:", bias)
    for xi in X:
        print(xi, "->", int(np.dot(xi, weights) + bias >= 1))
    plt.scatter(X[:,0], X[:,1], c=y)
    x = np.linspace(-0.2, 1.2, 100)
    if abs(weights[1]) > 1e-8:
        plt.plot(x, (1 - bias - weights[0]*x)/weights[1])
    plt.title("Perceptron AND gate")
    plt.show()
