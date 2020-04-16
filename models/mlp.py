import numpy as np
import utils
from sklearn.utils import shuffle
from numba import jit, cuda
from numpy.linalg import norm


# helper functions to transform between one big vector of weights
# and a list of layer parameters of the form (W,b)
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


def log_sum_exp(Z):
    Z_max = np.max(Z, axis=1)
    return Z_max + np.log(np.sum(np.exp(Z - Z_max[:, None]), axis=1))  # per-colmumn max


class NeuralNet():
    # uses sigmoid nonlinearity
    def __init__(self, hidden_layer_sizes, lammy=1, max_iter=500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lammy = lammy
        self.max_iter = max_iter

    def funObj(self, weights_flat, X, y):
        weights = unflatten_weights(weights_flat, self.layer_sizes)

        activations = [X]
        for W, b in weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
            activations.append(X)

        yhat = Z

        if self.classification:  # softmax- TODO: use logsumexp trick to avoid overflow
            tmp = np.sum(np.exp(yhat), axis=1)
            f = -np.sum(yhat[y.astype(bool)] - log_sum_exp(yhat))
            grad = np.exp(yhat) / tmp[:, None] - y
        else:  # L2 loss
            f = 0.5 * np.sum((yhat - y) ** 2)
            grad = yhat - y  # gradient for L2 loss

        grad_W = grad.T @ activations[-2]
        grad_b = np.sum(grad, axis=0)

        g = [(grad_W, grad_b)]

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            W, b = weights[i]
            grad = grad @ W
            grad = grad * (activations[i] * (1 - activations[i]))  # gradient of logistic loss
            grad_W = grad.T @ activations[i - 1]
            grad_b = np.sum(grad, axis=0)

            g = [(grad_W, grad_b)] + g  # insert to start of list

        g = flatten_weights(g)

        # add L2 regularization
        f += 0.5 * self.lammy * np.sum(weights_flat ** 2)
        g += self.lammy * weights_flat

        return f, g


    # # stochastic gradient descent
    # def fit(self, X, y):
    #     if y.ndim == 1:
    #         y = y[:, None]
    #     n, d = X.shape
    #     self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
    #     self.classification = y.shape[1] > 1  # assume it's classification iff y has more than 1 column
    #     # random init
    #     scale = 0.01
    #     weights = list()
    #     for i in range(len(self.layer_sizes) - 1):
    #         W = scale * np.random.randn(self.layer_sizes[i + 1], self.layer_sizes[i])
    #         b = scale * np.random.randn(1, self.layer_sizes[i + 1])
    #         weights.append((W, b))
    #     weights_flat = flatten_weights(weights)
    #     alpha = 0.001
    #     batch_size = 2500
    #     epochs = 250
    #     for epoch in range(epochs):
    #         X, y = shuffle(X, y)
    #         for i in range(0, n, batch_size):
    #             weights_flat, f = SGD(self.funObj, weights_flat, X[i:i + batch_size, :], y[i:i + batch_size, :],
    #                                           verbose=True, alpha=alpha)
    #     self.weights = unflatten_weights(weights_flat, self.layer_sizes)


    # regular gradient descent
    def fit(self, X, y):
        if y.ndim == 1:
            y = y[:,None]
        self.layer_sizes = [X.shape[1]] + self.hidden_layer_sizes + [y.shape[1]]
        self.classification = y.shape[1]>1 # assume it's classification iff y has more than 1 column
        # random init
        scale = 0.01
        weights = list()
        for i in range(len(self.layer_sizes)-1):
            W = scale * np.random.randn(self.layer_sizes[i+1],self.layer_sizes[i])
            b = scale * np.random.randn(1,self.layer_sizes[i+1])
            weights.append((W,b))
        weights_flat = flatten_weights(weights)
        weights_flat_new, f = findMin(self.funObj, weights_flat, self.max_iter, X, y, verbose=True)
        self.weights = unflatten_weights(weights_flat_new, self.layer_sizes)


    def predict(self, X):
        for W, b in self.weights:
            Z = X @ W.T + b
            X = 1 / (1 + np.exp(-Z))
        if self.classification:
            return np.argmax(Z, axis=1)
        else:
            return Z


def SGD(funObj, w, *args, verbose=0, alpha):
    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    w_new = w - alpha * g
    f_new, g_new = funObj(w_new, *args)
    # print("loss: %.3f" % f_new)

    # Update parameters/function/gradient
    w = w_new
    f = f_new

    return w, f


def findMin(funObj, w, maxEvals, *args, verbose=0):
    """
    Uses gradient descent to optimize the objective function
    This uses quadratic interpolation in its line search to
    determine the step size alpha
    """
    # Parameters of the Optimization
    optTol = 1e-2
    gamma = 1e-4

    # Evaluate the initial function value and gradient
    f, g = funObj(w,*args)
    funEvals = 1

    alpha = 1.
    while True:
        # Line-search using quadratic interpolation to
        # find an acceptable value of alpha
        gg = g.T.dot(g)

        while True:
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, *args)

            funEvals += 1
            if f_new <= f - gamma * alpha*gg:
                break

            if verbose > 1:
                print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # Update step size alpha
            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

        # Print progress
        # if verbose > 0:
            # print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        if verbose > 1:
            print("alpha: %.3f" % (alpha))

        # Update parameters/function/gradient
        w = w_new
        f = f_new
        g = g_new

        # Test termination conditions
        optCond = norm(g, float('inf'))

        if optCond < optTol:
            if verbose:
                print("Problem solved up to optimality tolerance %.3f" % optTol)
            break

        if funEvals >= maxEvals:
            if verbose:
                print("Reached maximum number of function evaluations %d" % maxEvals)
            break

    return w, f


