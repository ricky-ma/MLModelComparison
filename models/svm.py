import numpy as np
from sklearn.utils import shuffle
from optimization import findMin, SGD

class SVM():
    def __init__(self, lammy=1.0, alpha=0.01, maxEpochs=1000, batchSize=500):
        self.lammy = lammy
        self.alpha = alpha
        self.maxEpochs = maxEpochs
        self.batchSize = batchSize


    def funObj(self, w, X, y):
        N,D = X.shape

        # Calculate the function value (hinge loss)
        scores = np.dot(X,w)
        yi_scores = scores[np.arange(scores.shape[0]),y]
        margins = np.maximum(0, scores - np.matrix(yi_scores).T + 1)


        # margins = 1 - y + np.dot(X,w)
        # margins[margins < 0] = 0
        margins[np.arange(N), y] = 0
        f = np.mean(np.sum(margins, axis=1))
        f += 0.5 * self.lammy * np.sum(w**2) # Add L2 regularization

        # Calculate the gradient value
        res = margins
        res[margins > 0] = 1
        row_sum = np.sum(res, axis=1)
        res[np.arange(N), y] = -row_sum.T
        g = np.dot(X.T, res) / N
        g += self.lammy*w # Add L2 regularization

        return f, g


    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add bias variable
        self.weights = np.zeros([X.shape[1], 10])
        self.weights, f = SGD(self.funObj, self.weights, X, y, alpha=1., epochs=3, batch_size=2500)


    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add bias variable
        probabilities = np.dot(X, self.weights)
        predictions = np.argmax(probabilities, axis=1)
        return predictions


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
