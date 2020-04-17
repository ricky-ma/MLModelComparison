import os
import pickle
import gzip
import argparse
import numpy as np
import time

from sklearn.preprocessing import LabelBinarizer

from models import mlp, regression, knn, svm, pca


def load_dataset(filename):
    with open(os.path.join('..', 'data', filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-q', '--question', required=True)

    io_args = parser.parse_args()
    question = io_args.question


    if question == "knn":
        with gzip.open(os.path.join('mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        for k in [1,2,3]:
            print("k=%d" % k)
            start_time = time.time()
            model = knn.KNN(k=k)
            model.fit(X,y)
            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)
            y_pred = model.predict(Xtest)
            te_error = np.mean(y_pred != ytest)
            print("Training error k=1: %.4f" % tr_error)
            print("Testing error k=1: %.4f" % te_error)
            print("Runtime: %s seconds" % (time.time() - start_time))


    if question == "regression":
        with gzip.open(os.path.join('mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set
        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        X_subset = X[:100]
        y_subset = y[:100]

        # print("Fitting pricipal components")
        # model = pca.AlternativePCA(k=300)
        # model.fit(X)
        # print("Compressing")
        # Z = model.compress(X)
        # print("Expanding")
        # Xhat_pca = model.expand(Z)

        print("Fitting regression model")
        start_time = time.time()
        model = regression.logRegL2(verbose=True)
        model.fit(X,y)

        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        y_pred = model.predict(Xtest)
        te_error = np.mean(y_pred != ytest)

        print("Training error = %.4f" % tr_error)
        print("Testing error = %.4f" % te_error)
        print("Runtime: %s seconds" % (time.time() - start_time))


    if question == "svm":
        # TODO: fix error/reimplement
        with gzip.open(os.path.join('mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        start_time = time.time()
        model = svm.SVM()
        model.fit(X,y)
        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)
        y_pred = model.predict(Xtest)
        te_error = np.mean(y_pred != ytest)

        print("Training error = %.4f" % tr_error)
        print("Testing error = %.4f" % te_error)
        print("Runtime: %s seconds" % (time.time() - start_time))


    if question == "mlp":
        with gzip.open(os.path.join('mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)

        hidden_layer_sizes = [50,75,100,500]

        for size in hidden_layer_sizes:
            start_time = time.time()
            print("Hidden layer size: " + str(size))
            model = mlp.NeuralNet([size])
            model.fit(X,Y)

            yhat = model.predict(X)
            tr_error = np.mean(yhat != y)
            yhat = model.predict(Xtest)
            te_error = np.mean(yhat != ytest)
            print("Training error = %.4f" % tr_error)
            print("Testing error = %.4f" % te_error)
            print("Runtime: %s seconds" % (time.time() - start_time))
            print("\n")


    if question == "cnn":
        with gzip.open(os.path.join('..', 'data', 'mnist.pkl.gz'), 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
        X, y = train_set
        Xtest, ytest = test_set

        binarizer = LabelBinarizer()
        Y = binarizer.fit_transform(y)
        #TODO: convolution neural network (CNN)


    else:
        pass
