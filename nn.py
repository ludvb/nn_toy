#!/bin/env python3

# pylint: disable=invalid-name

import pdb

import matplotlib.pyplot as plt
import numpy as np


def gen_data(N, K, D):
    if N % K != 0:
        raise Exception("N must be divisible by K")

    X = np.zeros((N, D))
    Y = np.zeros((N, 1))

    for k in range(K):
        n = N / K
        r = np.linspace(0, 1, n)
        t = np.linspace(0, 2 * np.pi / K, n) + 2 * np.pi * k / K
        t += np.random.uniform(0, 2 * np.pi / 20, n)

        X[n * k:n * (k + 1), :] = np.array([r * np.cos(t), r * np.sin(t)]).T
        Y[n * k:n * (k + 1), 0] = k

    return X, Y


def main(N=300, K=3, D=2, nodes=100):
    X, Y = gen_data(N, K, D)
    W = np.uniform(-1, 1, D * nodes).reshape(nodes, D)

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()

if __name__ == "__main__":
    main()
