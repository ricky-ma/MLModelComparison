import numpy as np
import utils
from optimization import findMin, SGD


def log_1_plus_exp_safe(x):
    # compute log(1+exp(x)) in a numerically safe way, avoiding overflow/underflow issues
    out = np.log(1+np.exp(x))
    out[x > 100] = x[x>100]
    out[x < -100] = np.exp(x[x < -100])
    return out


def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z),axis=1)).T
    return sm


class Softmax():
    # L2 Regularized Multiclass Softmax Regression
    def __init__(self, lammy=1.0, verbose=0, maxEvals=150):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals


    def funObj(self, w, X, y):
        n = X.shape[0]  # First we get the number of training examples
        prob = softmax(np.dot(X, w))
        f = (-1 / n) * np.sum(y * np.log(prob))  # Calculate loss
        g = (-1 / n) * np.dot(X.T, (y - prob))   # Calculate gradient

        # # Add L2 regularization
        f += 0.5 * self.lammy * np.sum(w**2)
        g += self.lammy * w
        return f, g


    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add bias variable
        self.w = np.zeros([X.shape[1],10])
        # (self.w, f) = findMin(self.funObj, self.w, self.maxEvals, X, y, verbose=2)
        self.w, f = SGD(self.funObj, self.w, X, y, alpha=0.1, epochs=2, batch_size=5000)


    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add bias variable
        probabilities = softmax(np.dot(X, self.w))
        predictions = np.argmax(probabilities, axis=1)
        return predictions




