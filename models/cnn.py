import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
from optimization import adamGD


class CNN():
    def __init__(self, classes=10, alpha=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8,
              batch_size=32, epochs=2, save_path='params.pkl'):
        self.classes = classes
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.img_dim = img_dim
        self.img_depth = img_depth
        self.f = f
        self.num_filt1 = num_filt1
        self.num_filt2 = num_filt2
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_path = save_path


    def convolution(self, image, filt, bias, s=1):
        (n_f, n_c_f, f, _) = filt.shape     # filter dimensions
        n_c, in_dim, _ = image.shape        # image dimensions

        out_dim = int((in_dim - f) / s) + 1
        out = np.zeros((n_f, out_dim, out_dim))

        # convolve each filter over the image
        for curr_f in range(n_f):
            curr_y = out_y = 0
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                while curr_x + f <= in_dim:
                    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f])\
                                                + bias[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        return out


    def maxpool(self, image, f=2, s=2):
        n_c, h_prev, w_prev = image.shape

        # set output dimensions
        h = int((h_prev - f) / s) + 1
        w = int((w_prev - f) / s) + 1
        downsampled = np.zeros((n_c, h, w))

        # slide the window over every part of the image, taking the max value at each step
        for i in range(n_c):
            curr_y = out_y = 0
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                while curr_x + f <= w_prev:
                    downsampled[i, out_y, out_x] = np.max(image[i, curr_y:curr_y + f, curr_x:curr_x + f])
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
        return downsampled


    def softmax(self, raw_preds):
        out = np.exp(raw_preds)
        return out/np.sum(out)


    def initializeFilter(self, size, scale = 1.0):
        stddev = scale/np.sqrt(np.prod(size))
        return np.random.normal(loc = 0, scale = stddev, size = size)


    def convolutionBackward(self, dconv_prev, conv_in, filt, s):
        (n_f, n_c, f, _) = filt.shape
        (_, orig_dim, _) = conv_in.shape

        # initialize derivatives
        dout = np.zeros(conv_in.shape)
        dfilt = np.zeros(filt.shape)
        dbias = np.zeros((n_f, 1))

        for curr_f in range(n_f):
            # loop through all filters, calculating loss gradients for filter, input, and bias
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
                    dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            dbias[curr_f] = np.sum(dconv_prev[curr_f])
        return dout, dfilt, dbias


    def maxpoolBackward(self, dpool, orig, f, s):
        (n_c, orig_dim, _) = orig.shape
        dout = np.zeros(orig.shape)

        for curr_c in range(n_c):
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    arr = orig[curr_c, curr_y:curr_y + f, curr_x:curr_x + f]
                    idx = np.nanargmax(arr)
                    (a,b) = np.unravel_index(idx, arr.shape)
                    dout[curr_c, curr_y + a, curr_x + b] = dpool[curr_c, out_y, out_x]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1

        return dout


    def conv(self, image, label, params, conv_s, pool_f, pool_s):
        [f1, f2, w3, w4, b1, b2, b3, b4] = params

        # forward
        conv1 = self.convolution(image, f1, b1, conv_s)  # first convolution operation
        conv1[conv1 <= 0] = 0  # ReLU activation

        conv2 = self.convolution(conv1, f2, b2, conv_s)  # second convolution operation
        conv2[conv2 <= 0] = 0  # ReLU activation

        pooled = self.maxpool(conv2, pool_f, pool_s)  # maxpooling operation
        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1))

        z = w3.dot(fc) + b3  # first dense layer
        z[z <= 0] = 0  # ReLU activation
        out = w4.dot(z) + b4  # second dense layer

        probs = self.softmax(out)  # predict class probabilities with softmax
        loss = -np.sum(label * np.log(probs))  # categorical cross-entropy loss

        # backward
        # calculate gradients of second dense layer
        dout = probs - label  # derivative of loss w.r.t. final dense layer output
        dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
        db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases

        # calculate gradients of first dense layer
        dz = w4.T.dot(dout)
        dz[z <= 0] = 0  # backpropagate through ReLU activation
        dw3 = dz.dot(fc.T)
        db3 = np.sum(dz, axis=1).reshape(b3.shape)

        # calculate gradient of pooling layer
        dfc = w3.T.dot(dz)
        dpool = dfc.reshape(pooled.shape)

        dconv2 = self.maxpoolBackward(dpool, conv2, pool_f, pool_s)  # backpropagate through pooling layer
        dconv2[conv2 <= 0] = 0  # backpropagate through ReLU activation

        dconv1, df2, db2 = self.convolutionBackward(dconv2, conv1, f2, conv_s)  # backpropagate through second convolutional layer
        dconv1[conv1 <= 0] = 0  # backpropagate through ReLU activation

        dimage, df1, db1 = self.convolutionBackward(dconv1, image, f1, conv_s)  # backpropagate through first convolutional layer

        grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]
        return grads, loss


    def fit(self, X, y):
        train_data = np.hstack((X, y))
        np.random.shuffle(train_data)

        # initialize parameters
        f1, f2, w3, w4 = (self.num_filt1, self.img_depth, self.f, self.f), \
                         (self.num_filt2, self.num_filt1, self.f, self.f), (128, 800), (10, 128)
        f1 = self.initializeFilter(f1)
        f2 = self.initializeFilter(f2)
        w3 = np.random.standard_normal(w3) * 0.01
        w4 = np.random.standard_normal(w4) * 0.01

        b1 = np.zeros((f1.shape[0], 1))
        b2 = np.zeros((f2.shape[0], 1))
        b3 = np.zeros((w3.shape[0], 1))
        b4 = np.zeros((w4.shape[0], 1))

        params = [f1, f2, w3, w4, b1, b2, b3, b4]
        cost = []

        print("Alpha:" + str(self.alpha) + ", Batch Size:" + str(self.batch_size))

        # print training progress while optimizing with adamGD
        for epoch in range(self.epochs):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + self.batch_size] for k in range(0, train_data.shape[0], self.batch_size)]

            t = tqdm(batches)
            for x, batch in enumerate(t):
                params, cost = adamGD(batch, self.classes, self.alpha, self.img_dim, self.img_depth,
                                      self.beta1, self.beta2, params, cost, self.conv)
                t.set_description("Cost: %.2f" % (cost[-1]))

        # save parameters (weights)
        with open(self.save_path, 'wb') as file:
            pickle.dump(params, file)

        return cost


    def predict(self, X, conv_s=1, pool_f=2, pool_s=2):
        with open('params.pkl', 'rb') as f:
            params = pickle.load(f)
        [f1, f2, w3, w4, b1, b2, b3, b4] = params

        X = X.reshape(len(X), 1, 28, 28)

        predictions = np.array([])
        img_counter = 1
        for image in X:
            print("Predicting image: " + str(img_counter))
            conv1 = self.convolution(image, f1, b1, conv_s)
            conv1[conv1 <= 0] = 0

            conv2 = self.convolution(conv1, f2, b2, conv_s)
            conv2[conv2 <= 0] = 0

            pooled = self.maxpool(conv2, pool_f, pool_s)
            (nf2, dim2, _) = pooled.shape
            fc = pooled.reshape((nf2 * dim2 * dim2, 1))

            z = w3.dot(fc) + b3
            z[z <= 0] = 0

            out = w4.dot(z) + b4
            probs = self.softmax(out)

            predictions = np.append(predictions, np.argmax(probs))
            img_counter += 1
        return predictions.reshape(len(X), 1)


