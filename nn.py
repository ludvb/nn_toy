#!/bin/env python3

"""Some module docstring"""

# pylint: disable=invalid-name, no-member

import pdb

import matplotlib.pyplot as plt
import numpy as np

import layers as L


def gen_data(N, K, D):
    """Generate the data"""
    if N % K != 0:
        raise Exception("N must be divisible by K")

    X = np.zeros((N, D))
    Y = np.zeros((N, 1), dtype=int)

    for k in range(K):
        n = int(N / K)
        r = np.linspace(0.01, 1, n)
        t = np.linspace(0, 2 * 2 * np.pi / K, n) + 2 * np.pi * k / K
        t += np.random.uniform(0, 2 * np.pi / 6, n)

        idx = slice(n * k, n * (k + 1))
        X[idx, :] = np.transpose(
            np.array([r * np.cos(t), r * np.sin(t)]))
        Y[idx, 0] = k

    return X, Y


def main(N=300, K=3, D=2, nodes=100, reg=1e-3):
    """Main"""
    # Generate and plot data set
    X, Y = gen_data(N, K, D)

    print("Plotting data...")
    plt.ion()
    plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.winter)
    plt.draw()

    input("Press <ENTER> to continue.")

    # Set up layers
    layers = []
    layers += [L.input(X)]
    layers += [L.fc(layers[-1].Y, 50)]
    layers += [L.sigmoid(layers[-1].Y)]
    layers += [L.fc(layers[-1].Y, 50)]
    layers += [L.sigmoid(layers[-1].Y)]
    layers += [L.fc(layers[-1].Y, 4)]
    layers += [L.sigmoid(layers[-1].Y)]
    layers += [L.softmax(layers[-1].Y)]
    layers += [L.loss(layers[-1].Y, Y)]

    nlayers = len(layers)

    # TODO (architecture): Instead of calling fwd on each layer, connect layers
    # with "pointers" and call fwd only on the first layer

    try:
        itx = 1
        while True:
            # Forward propagation
            for i, layer in enumerate(layers):
                if i == 0:
                    layer.X = X
                else:
                    layer.reshape(layers[i - 1].Y.shape)
                    layer.X = layers[i - 1].Y
                layer.fwd()

            if np.isnan(layers[-1].Y[0, 0]):
                pdb.set_trace()

            print("Iteration {}, Loss = {:.4f}"
                  .format(itx, np.asscalar(layers[-1].Y)))

            # Backprop
            for i in list(range(nlayers))[::-1]:
                if i == nlayers - 1:
                    layers[i].dy = 1
                else:
                    layers[i].dy = layers[i + 1].dx

                layers[i].bck()
                layers[i].step(1e-3, 1e-8)

            if itx % 5000 == 0:
                range_ = [np.max(X[:, i]) - np.min(X[:, i]) for i in (0, 1)]
                x, y = [
                    np.linspace(np.min(X[:, i]) - range_[i]/2,
                                np.max(X[:, i]) + range_[i]/2,
                                400)
                    for i in (0, 1)
                ]
                xx, yy = np.meshgrid(x, y)

                X_ = np.c_[xx.flatten(), yy.flatten()]
                for i, layer in enumerate(layers[:-1]):
                    if i == 0:
                        layer.X = X_
                    else:
                        layer.reshape(layers[i - 1].Y.shape)
                        layer.X = layers[i - 1].Y
                    layer.fwd()
                z = np.argmax(layers[-2].Y, axis=1).reshape(xx.shape)

                plt.clf()
                plt.contourf(xx, yy, z, cmap=plt.cm.winter)
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.winter)
                plt.draw()
                plt.pause(1e-10)

            if False:  # Gradient check
                for i in list(range(nlayers))[::-1]:
                    if np.all(layers[i].dx == 0):
                        continue

                    print("Checking gradient on {}".format(layers[i]))
                    r, c = [
                        np.random.choice(layers[i].X.shape[j])
                        for j in (0, 1)
                    ]
                    h = 1e-8

                    Y_ = []
                    for X_ in [
                            layers[i].X[r, c] + s * h
                            for s in (-1, 1)
                    ]:
                        layers[i].X[r, c] = X_

                        for j in range(i, nlayers):
                            if j > i:
                                layers[j].reshape(layers[j - 1].Y.shape)
                                layers[j].X = layers[j - 1].Y
                            layers[j].fwd()

                        Y_.append(np.asscalar(layers[-1].Y))

                    ndx = (Y_[1] - Y_[0]) / (2 * h)
                    print("Analytical: {}, Numerical: {}".format(
                        layers[i].dx[r, c], ndx
                    ))
                    input("")

            itx += 1

    except KeyboardInterrupt:
        # print(layers[-2].Y)
        pass


if __name__ == "__main__":
    main(300, 4, 2, 5)
