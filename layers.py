"""Specifies the layer types in the net"""

import numpy as np

# pylint:disable=invalid-name, redefined-builtin

class Layer(object):
    """Layer class. All layers inherit from this class."""
    Y, dy = None, None
    W, dw = None, None

    stochastic = False

    def __init__(self, X):
        self.X = X
        self.dx = np.zeros_like(X)
        self.fwd()

    def fwd(self):
        """Abstract method: forward pass"""
        raise NotImplementedError(
            "{} not fully implemented".format(self))

    def bck(self):
        """Abstract method: backward pass"""
        raise NotImplementedError(
            "{} not fully implemented".format(self))

    def step(self, lr, reg):
        """Abstract method: parameter update"""
        raise NotImplementedError(
            "{} not fully implemented".format(self))

    def reshape(self, shape):
        """Reshapes input to the given shape"""
        pass


class input(Layer):
    """Input layer"""

    def fwd(self):
        self.Y = self.X

    def bck(self):
        pass

    def step(self, lr, reg):
        pass


class fc(Layer):
    """Fully connected layer. N = #(nodes)"""

    def __init__(self, X, N):
        self.N = N
        self.i = X.shape[1]  # #(inputs)

        self.dw = None
        self.reshape(X.shape)

        # Init weights
        self.W = np.random.uniform(
            -1, 1, (self.i + 1) * self.N
        ).reshape(self.i + 1, self.N)

        super().__init__(X)

    def fwd(self):
        self.Y = np.dot(self._x_bias(), self.W)

    def bck(self):
        self.dx = np.dot(self.dy, self.W.T[:, :-1])
        self.dw = np.dot(self._x_bias().T, self.dy)

    def step(self, lr, reg):
        self.W -= lr * self.dw + 2 * reg * self.W

    def reshape(self, shape):
        self.n = shape[0]  # batch size

        # Init output
        self.Y = np.zeros((self.n, self.N))

    def _x_bias(self):
        """Add bias terms to input"""
        bias_terms = np.array([1] * self.n)[:, np.newaxis]
        return np.concatenate((self.X, bias_terms), 1)


class relu(Layer):
    """Rectified linear unit"""

    def fwd(self):
        self.Y = self.X * (self.X > 0)

    def bck(self):
        self.dx = self.dy * (self.Y > 0)

    def step(self, lr, reg):
        pass


class sigmoid(Layer):
    """Sigmoid unit"""

    def fwd(self):
        self.Y = 1 / (1 + np.exp(-self.X))

    def bck(self):
        self.dx = self.dy * self.Y * (1 - self.Y)

    def step(self, lr, reg):
        pass


class dropout(Layer):
    """Dropout layer"""

    stochastic = True

    def __init__(self, X, prob):
        self.p = prob
        self.reshape(X.shape)
        super().__init__(X)

    def fwd(self):
        if self.stochastic:
            self.mask = (np.random.uniform(size=self.X.shape)
                         > self.p) / (1 - self.p)
        self.Y = self.X * self.mask

    def bck(self):
        self.dx = self.dy * self.mask

    def step(self, lr, reg):
        pass

    def reshape(self, shape):
        self.mask = np.ones(shape)


class softmax(Layer):
    """Softmax layer"""

    def __init__(self, X):
        self.reshape(X.shape)
        super().__init__(X)

    def fwd(self):
        self.Y[...] = np.exp(self.X)
        self.Y[...] = np.einsum(
            'ij,i->ij', self.Y, 1 / np.sum(self.Y, 1)
        )

    def bck(self):
        dyY = self.dy * self.Y
        dyYs = np.sum(dyY, 1)
        self.dx = self.Y * (self.dy.T - dyYs).T

    def step(self, lr, reg):
        pass

    def reshape(self, shape):
        self.Y = np.zeros(shape)


class loss(Layer):
    """Loss layer"""

    def __init__(self, X, X_labels):
        self.Y = np.array([[0.]])
        self.X_labels = X_labels
        super().__init__(X)

    def fwd(self):
        probs = self.X[
            range(len(self.X[:, 0])),
            self.X_labels[:, 0]
        ]
        self.Y[...] = np.sum(-np.log(probs))

    def bck(self):
        labels = list(range(len(self.X[0, :])))
        mlabels = np.array([labels] * len(self.X[:, 0]))
        self.dx = self.dy * (-1) / (self.X) * (mlabels == self.X_labels)

    def step(self, lr, reg):
        pass


class sum(Layer):
    """Summing operator"""

    def __init__(self, X):
        self.Y = np.array([[0.]])
        super().__init__(X)

    def fwd(self):
        self.Y[...] = np.sum(self.X)

    def bck(self):
        self.dx = np.ones_like(self.X)

    def step(self, lr, reg):
        pass
