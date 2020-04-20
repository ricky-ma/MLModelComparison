import numpy as np
from sklearn.utils import shuffle
from optimization import findMin, SGD
from utils import flatten_weights, unflatten_weights

class SVM():
    def __init__(self, lammy=1.0, alpha=0.01, epochs=1000, batchSize=500):
        self.lammy = lammy
        self.alpha = alpha
        self.epochs = epochs
        self.batchSize = batchSize


    def funObj(self, w, X, y):
        N,D = X.shape

        f = 0
        g = np.zeros([X.shape[1],10])
        for i in range(N):
            scores = np.dot(X[i], w)
            label = scores[y[i]]
            for j in range(10):
                margin = 1 + scores[j] - label
                if y[i] != j and margin > 0:
                    f += margin
                    g[:,y[i]] -= X[i,:]
                    g[:,j] += X[i,:]
        f /= N
        g /= N
        f += 0.5 * self.lammy * np.sum(w**2)
        g += self.lammy * w # Add L2 regularization
        return f, g


    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add bias variable
        self.weights = np.zeros([X.shape[1],10])
        self.weights, f = SGD(self.funObj, self.weights, X, y, alpha=self.alpha, epochs=self.epochs, batch_size=self.batchSize)
        # self.weights, f = findMin(self.funObj, self.weights, X, y, maxEvals=1000, verbose=2)


    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add bias variable
        probabilities = np.dot(X, self.weights)
        print(probabilities)
        predictions = np.argmax(probabilities, axis=1)
        print(predictions)
        return predictions
