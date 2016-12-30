#!/bin/env python3

"""Some module docstring"""

# pylint: disable=invalid-name, no-member

import pdb

import matplotlib.pyplot as plt
import matplotlib.colors as col
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
        r = np.linspace(0.02, 1, n)
        t = np.linspace(0, 1.5 * K * 2 * np.pi / K, n) + 2 * np.pi * k / K
        t += np.random.uniform(0, 2 * np.pi / (8 * K), n)

        idx = slice(n * k, n * (k + 1))
        X[idx, :] = np.transpose(
            np.array([r * np.cos(t), r * np.sin(t)]))
        Y[idx, 0] = k

    return X, Y


def main(N=300, K=3, D=2, nodes=100, lr=1e-3, reg=1e-8):
    """Main"""
    # Generate and plot data set
    X, Y = gen_data(N, K, D)

    print("Plotting data...")
    col_levels = np.array(list(range(K + 1)), dtype=np.float) - 0.5
    col_cmap = plt.cm.gist_rainbow
    col_norm = col.BoundaryNorm(col_levels, col_cmap.N)

    plt.ion()
    plt.subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=col_cmap, norm=col_norm,
                vmin=np.min(Y), vmax=np.max(Y))
    plt.draw()

    input("Press <ENTER> to continue.")

    # Set up layers
    layers = []
    layers += [L.input(X)]
    layers += [L.fc(layers[-1].Y, nodes)]
    layers += [L.sigmoid(layers[-1].Y)]
    layers += [L.dropout(layers[-1].Y, 0.25)]
    layers += [L.fc(layers[-1].Y, nodes)]
    layers += [L.sigmoid(layers[-1].Y)]
    layers += [L.dropout(layers[-1].Y, 0.25)]
    layers += [L.fc(layers[-1].Y, K)]
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
                  .format(itx, np.asscalar(layers[-1].Y)),
                  end='\r')

            if itx % 1000 == 0:
                print("")

            # Backprop
            for i in list(range(nlayers))[::-1]:
                if i == nlayers - 1:
                    layers[i].dy = 1
                else:
                    layers[i].dy = layers[i + 1].dx

                layers[i].bck()

                if itx % 5000 == 0:  # Gradient check
                    if np.all(layers[i].dx == 0):
                        continue

                    r, c = [
                        np.random.choice(layers[i].X.shape[j])
                        for j in (0, 1)
                    ]
                    h = 1e-4

                    if abs(layers[i].dx[r, c]) < 1e-5:
                        continue

                    print("Checking gradient on {}...".format(layers[i]),
                          end=' ')

                    X_store = layers[i].X

                    Y_ = []
                    for X_ in [
                            layers[i].X[r, c] + s * h
                            for s in (-1, 1)
                    ]:
                        layers[i].X[r, c] = X_

                        for j in range(i, nlayers):
                            if j > i:
                                layers[j].X = layers[j - 1].Y

                            stochastic_store = layers[j].stochastic
                            layers[j].stochastic = False
                            layers[j].fwd()
                            layers[j].stochastic = stochastic_store

                        Y_.append(np.asscalar(layers[-1].Y))

                    layers[i].X = X_store

                    dx = layers[i].dx[r, c]
                    ndx = (Y_[1] - Y_[0]) / (2 * h)
                    diff = abs(ndx - dx) / max(abs(ndx), abs(dx), 1e-10)

                    print("Diff: {:.8f}".format(diff))
                    if diff > 1e-2:
                        pdb.set_trace()

            for layer in layers:
                layer.step(lr, reg)

            if itx % 1000 == 0:
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

                    temp = layer.stochastic
                    layer.stochastic = False
                    layer.fwd()
                    layer.stochastic = temp

                z = np.argmax(layers[-2].Y, axis=1).reshape(xx.shape)

                plt.clf()
                plt.contourf(xx, yy, z, levels=col_levels, cmap=col_cmap,
                             norm=col_norm)
                plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=col_cmap,
                            norm=col_norm)
                plt.draw()
                plt.pause(1e-10)

            itx += 1

    except KeyboardInterrupt:
        # print(layers[-2].Y)
        pass


if __name__ == "__main__":
    main(1000, 5, 2, 100, 1e-3, 1e-8)
