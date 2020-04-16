import numpy as np
from sklearn.utils import shuffle

class SVM():
    def __init__(self, lammy=1.0, alpha=0.0001, maxEpochs=1000, batchSize=500):
        self.lammy = lammy
        self.alpha = alpha
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize


    def funObj(self, w, X, y):
        N,D = X.shape

        # Calculate the function value (hinge loss)
        scores = X.dot(w)
        labels = scores[np.arange(N), y]
        margins = np.max(0, 1. - labels[:,np.newaxis] + X.dot(w))
        margins[np.arange(N), y] = 0
        f = np.sum(margins) / N
        f += 0.5 * self.lammy * np.sum(w*w) # Add L2 regularization

        # Calculate the gradient value
        res = np.zeros(margins.shape)
        res[margins > 0] = 1
        res[np.arange(N),y] = -np.sum(res, axis=1)
        g = X.T.dot(res) / N
        g += self.lammy*w # Add L2 regularization

        return f, g


    def fit(self, X, y):
        N,D = X.shape
        w = np.zeros((D, 10))
        for epoch in range(1, self.maxEpochs):
            X,y = shuffle(X,y)
            w, f = SGD(self.funObj, w, X, y, verbose=True, alpha=self.alpha)

        self.weights = w


    def predict(self, X):
        y = np.dot(X, self.weights.T)
        prediction = y.argmax(axis=1)
        print(y)
        return prediction


def SGD(funObj, w, *args, verbose=0, alpha):
    # Evaluate the initial function value and gradient
    f, g = funObj(w, *args)
    w_new = w - alpha * g
    f_new, g_new = funObj(w_new, *args)
    print("loss: %.3f" % f_new)

    # Update parameters/function/gradient
    w = w_new
    f = f_new
    return w, f


def flatten_weights(weights):
    return np.concatenate([w.flatten() for w in sum(weights, ())])


def unflatten_weights(weights_flat, layer_sizes):
    weights = list()
    counter = 0
    for i in range(len(layer_sizes) - 1):
        W_size = layer_sizes[i + 1] * layer_sizes[i]
        b_size = layer_sizes[i + 1]

        W = np.reshape(weights_flat[counter:counter + W_size], (layer_sizes[i + 1], layer_sizes[i]))
        counter += W_size

        b = weights_flat[counter:counter + b_size][None]
        counter += b_size

        weights.append((W, b))
    return weights
