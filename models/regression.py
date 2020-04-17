import numpy as np
import utils
from optimization import findMin, SGD


def log_1_plus_exp_safe(x):
    # compute log(1+exp(x)) in a numerically safe way, avoiding overflow/underflow issues
    out = np.log(1+np.exp(x))
    out[x > 100] = x[x>100]
    out[x < -100] = np.exp(x[x < -100])
    return out


def kernel_poly(X1, X2, p=9):
    return (1+np.dot(X1,X2.T))**p


class logRegL2():
    # L2 Regularized Logistic Regression (no intercept)
    def __init__(self, lammy=1.0, verbose=0, maxEvals=150, alpha = 0.001):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.alpha = alpha

    def funObj(self, w, X, y):
        yXw = y * X.dot(w)
        # f = np.sum(np.log(1. + np.exp(-yXw))) # Calculate the function value
        f = np.sum(log_1_plus_exp_safe(-yXw))
        f += 0.5 * self.lammy * np.sum(w**2)    # Add L2 regularization
        res = - y / (1. + np.exp(yXw))          # Calculate the gradient value
        g = X.T.dot(res) + self.lammy * w
        return f, g

    def fit(self,X, y):
        # add intercept
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        n, d = X.shape
        # initialize weights
        self.w = np.zeros(d)
        # optimize
        # utils.check_gradient(self, X, y, d, verbose=self.verbose)
        # self.w, f = findMin(self.funObj, np.zeros(d), self.maxEvals, X, y, verbose=2)
        self.w, f = SGD(self.funObj, self.w, X, y)

    def predict(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        return np.sign(X@self.w)


class kernelLogRegL2():
    def __init__(self, lammy=1.0, verbose=0, maxEvals=100, kernel_fun=kernel_poly, **kernel_args):
        self.verbose = verbose
        self.lammy = lammy
        self.maxEvals = maxEvals
        self.kernel_fun = kernel_fun
        self.kernel_args = kernel_args


    def funObj(self, u, K, y):
        yKu = y * (K@u)
        # f = np.sum(np.log(1. + np.exp(-yKu)))
        f = np.sum(log_1_plus_exp_safe(-yKu))   # Calculate the function value
        f += 0.5 * self.lammy * u.T@K@u         # Add L2 regularization
        res = - y / (1. + np.exp(yKu))          # Calculate the gradient value
        g = (K.T@res) + self.lammy * K@u
        return f, g


    def fit(self, X, y):
        n, d = X.shape
        print("Calculating kernel")
        K = self.kernel_fun(X,X, **self.kernel_args)
        # print("Checking gradient")
        # utils.check_gradient(self, K, y, n, verbose=self.verbose)
        print("Optimizing")
        self.u, f = findMin(self.funObj, np.zeros(n), self.maxEvals, K, y, verbose=self.verbose)


    def predict(self, Xtest):
        Ktest = self.kernel_fun(Xtest, self.X, **self.kernel_args)
        return np.sign(Ktest@self.weights)



