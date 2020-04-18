import numpy as np
from numpy.linalg import norm, solve
from sklearn.utils import shuffle

# stochastic gradient descent
def SGD(funObj, w, X, y, *args, alpha=1, epochs=10, batch_size=10000):
    for epoch in range(1, epochs+1):
        alpha = step_decay(epoch)
        print("epoch: %.0f" % epoch)
        X, y = shuffle(X, y)
        for i in range(0, X.shape[0], batch_size):
            if (epoch - 1 % 5 == 0): print("alpha: %.0f" % alpha)
            # Evaluate the initial function value and gradient
            f, g = funObj(w, X, y, *args)
            w_new = w - alpha * g
            f_new, g_new = funObj(w_new, X, y, *args)
            print("loss: %.3f" % f_new)
            # Update parameters/function/gradient
            w = w_new
            f = f_new
    return w, f


def step_decay(epoch):
   initial_alpha = 0.1
   drop = 0.5
   epochs_drop = 5.0
   alpha = initial_alpha * np.power(drop, np.floor((1+epoch)/epochs_drop))
   return alpha


# standard gradient descent, quadratic interpolation to determine alpha
def findMin(funObj, w, maxEvals, *args, verbose=0):
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

            # if verbose > 1:
            print("f_new: %.3f - f: %.3f - Backtracking..." % (f_new, f))

            # Update step size alpha
            alpha = (alpha**2) * gg/(2.*(f_new - f + alpha*gg))

        # Print progress
        # if verbose > 0:
        print("%d - loss: %.3f" % (funEvals, f_new))

        # Update step-size for next iteration
        y = g_new - g
        alpha = -alpha*np.dot(y.T,g) / np.dot(y.T,y)

        # Safety guards
        if np.isnan(alpha) or alpha < 1e-10 or alpha > 1e10:
            alpha = 1.

        # if verbose > 1:
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