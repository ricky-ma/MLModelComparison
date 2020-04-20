import numpy as np
import pickle
import gzip
import matplotlib.pyplot as plt
from tqdm import tqdm
# from optimization import findMin, SGD


class CNN():
    def __init__(self, num_classes=10, alpha=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8,
              batch_size=32, num_epochs=2, save_path='params.pkl'):
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.img_dim = img_dim
        self.img_depth = img_depth
        self.f = f
        self.num_filt1 = num_filt1
        self.num_filt2 = num_filt2
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.save_path = save_path


    def convolution(self, image, filt, bias, s=1):
        (n_f, n_c_f, f, _) = filt.shape  # filter dimensions
        n_c, in_dim, _ = image.shape  # image dimensions

        out_dim = int((in_dim - f) / s) + 1  # calculate output dimensions

        # ensure that the filter dimensions match the dimensions of the input image
        assert n_c == n_c_f, "Dimensions of filter must match dimensions of input image"

        out = np.zeros((n_f, out_dim, out_dim))  # create the matrix to hold the values of the convolution operation

        # convolve each filter over the image
        for curr_f in range(n_f):
            curr_y = out_y = 0
            # move filter vertically across the image
            while curr_y + f <= in_dim:
                curr_x = out_x = 0
                # move filter horizontally across the image
                while curr_x + f <= in_dim:
                    # perform the convolution operation and add the bias
                    out[curr_f, out_y, out_x] = np.sum(filt[curr_f] * image[:, curr_y:curr_y + f, curr_x:curr_x + f]) + \
                                                bias[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1

        return out


    def maxpool(self, image, f=2, s=2):
        n_c, h_prev, w_prev = image.shape

        # calculate output dimensions after the maxpooling operation.
        h = int((h_prev - f) / s) + 1
        w = int((w_prev - f) / s) + 1

        # create a matrix to hold the values of the maxpooling operation.
        downsampled = np.zeros((n_c, h, w))

        # slide the window over every part of the image using stride s. Take the maximum value at each step.
        for i in range(n_c):
            curr_y = out_y = 0
            # slide the max pooling window vertically across the image
            while curr_y + f <= h_prev:
                curr_x = out_x = 0
                # slide the max pooling window horizontally across the image
                while curr_x + f <= w_prev:
                    # choose the maximum value within the window at each step and store it to the output matrix
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
        ## initialize derivatives
        dout = np.zeros(conv_in.shape)
        dfilt = np.zeros(filt.shape)
        dbias = np.zeros((n_f, 1))
        for curr_f in range(n_f):
            # loop through all filters
            curr_y = out_y = 0
            while curr_y + f <= orig_dim:
                curr_x = out_x = 0
                while curr_x + f <= orig_dim:
                    # loss gradient of filter (used to update the filter)
                    dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y + f, curr_x:curr_x + f]
                    # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                    dout[:, curr_y:curr_y + f, curr_x:curr_x + f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f]
                    curr_x += s
                    out_x += 1
                curr_y += s
                out_y += 1
            # loss gradient of the bias
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
        conv1 = self.convolution(image, f1, b1, conv_s)  # convolution operation
        conv1[conv1 <= 0] = 0  # pass through ReLU non-linearity

        conv2 = self.convolution(conv1, f2, b2, conv_s)  # second convolution operation
        conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

        pooled = self.maxpool(conv2, pool_f, pool_s)  # maxpooling operation

        (nf2, dim2, _) = pooled.shape
        fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

        z = w3.dot(fc) + b3  # first dense layer
        z[z <= 0] = 0  # pass through ReLU non-linearity

        out = w4.dot(z) + b4  # second dense layer

        probs = self.softmax(out)  # predict class probabilities with the softmax activation function

        loss = -np.sum(label * np.log(probs))  # categorical cross-entropy loss

        # backward
        dout = probs - label  # derivative of loss w.r.t. final dense layer output
        dw4 = dout.dot(z.T)  # loss gradient of final dense layer weights
        db4 = np.sum(dout, axis=1).reshape(b4.shape)  # loss gradient of final dense layer biases

        dz = w4.T.dot(dout)  # loss gradient of first dense layer outputs
        dz[z <= 0] = 0  # backpropagate through ReLU
        dw3 = dz.dot(fc.T)
        db3 = np.sum(dz, axis=1).reshape(b3.shape)

        dfc = w3.T.dot(dz)  # loss gradients of fully-connected layer (pooling layer)
        dpool = dfc.reshape(pooled.shape)  # reshape fully connected into dimensions of pooling layer

        dconv2 = self.maxpoolBackward(dpool, conv2, pool_f, pool_s)  # backprop through the max-pooling layer(only neurons with highest activation in window get updated)
        dconv2[conv2 <= 0] = 0  # backpropagate through ReLU

        dconv1, df2, db2 = self.convolutionBackward(dconv2, conv1, f2, conv_s)  # backpropagate previous gradient through second convolutional layer.
        dconv1[conv1 <= 0] = 0  # backpropagate through ReLU

        dimage, df1, db1 = self.convolutionBackward(dconv1, image, f1, conv_s)  # backpropagate previous gradient through first convolutional layer.

        grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

        return grads, loss


    def fit(self, X, y):
        X -= int(np.mean(X))
        X /= int(np.std(X))
        train_data = np.hstack((X, y))

        np.random.shuffle(train_data)

        ## Initializing all the parameters
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

        for epoch in range(self.num_epochs):
            np.random.shuffle(train_data)
            batches = [train_data[k:k + self.batch_size] for k in range(0, train_data.shape[0], self.batch_size)]

            t = tqdm(batches)
            for x, batch in enumerate(t):
                params, cost = self.adamGD(batch, self.num_classes, self.alpha, self.img_dim, self.img_depth, self.beta1, self.beta2, params, cost)
                t.set_description("Cost: %.2f" % (cost[-1]))

        with open(self.save_path, 'wb') as file:
            pickle.dump(params, file)

        return cost


    def predict(self, X, conv_s=1, pool_f=2, pool_s=2):
        with open('params.pkl', 'rb') as f:
            params = pickle.load(f)
        [f1, f2, w3, w4, b1, b2, b3, b4] = params

        X -= int(np.mean(X))  # subtract mean
        X /= int(np.std(X))  # divide by standard deviation
        # test_data = np.hstack((X, y))

        # X = test_data[:, 0:-1]
        X = X.reshape(len(X), 1, 28, 28)
        # y = test_data[:, -1]

        predictions = np.array([])
        img_counter = 1

        for image in X:
            print("Predicting image: " + str(img_counter))
            conv1 = self.convolution(image, f1, b1, conv_s)  # convolution operation
            conv1[conv1 <= 0] = 0  # relu activation

            conv2 = self.convolution(conv1, f2, b2, conv_s)  # second convolution operation
            conv2[conv2 <= 0] = 0  # pass through ReLU non-linearity

            pooled = self.maxpool(conv2, pool_f, pool_s)  # maxpooling operation
            (nf2, dim2, _) = pooled.shape
            fc = pooled.reshape((nf2 * dim2 * dim2, 1))  # flatten pooled layer

            z = w3.dot(fc) + b3  # first dense layer
            z[z <= 0] = 0  # pass through ReLU non-linearity

            out = w4.dot(z) + b4  # second dense layer
            probs = self.softmax(out)  # predict class probabilities with the softmax activation function

            predictions = np.append(predictions, np.argmax(probs))
            img_counter += 1

            # return np.argmax(probs), np.max(probs)
        return predictions.reshape(len(X), 1)


    def adamGD(self, batch, num_classes, alpha, dim, n_c, beta1, beta2, params, cost):
        [f1, f2, w3, w4, b1, b2, b3, b4] = params

        X = batch[:, 0:-1]  # get batch inputs
        X = X.reshape(len(batch), n_c, dim, dim)
        Y = batch[:, -1]  # get batch labels

        cost_ = 0
        batch_size = len(batch)

        # initialize gradients and momentum,RMS params
        df1 = np.zeros(f1.shape)
        df2 = np.zeros(f2.shape)
        dw3 = np.zeros(w3.shape)
        dw4 = np.zeros(w4.shape)
        db1 = np.zeros(b1.shape)
        db2 = np.zeros(b2.shape)
        db3 = np.zeros(b3.shape)
        db4 = np.zeros(b4.shape)

        v1 = np.zeros(f1.shape)
        v2 = np.zeros(f2.shape)
        v3 = np.zeros(w3.shape)
        v4 = np.zeros(w4.shape)
        bv1 = np.zeros(b1.shape)
        bv2 = np.zeros(b2.shape)
        bv3 = np.zeros(b3.shape)
        bv4 = np.zeros(b4.shape)

        s1 = np.zeros(f1.shape)
        s2 = np.zeros(f2.shape)
        s3 = np.zeros(w3.shape)
        s4 = np.zeros(w4.shape)
        bs1 = np.zeros(b1.shape)
        bs2 = np.zeros(b2.shape)
        bs3 = np.zeros(b3.shape)
        bs4 = np.zeros(b4.shape)

        for i in range(batch_size):
            x = X[i]
            y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot

            # Collect Gradients for training example
            grads, loss = self.conv(x, y, params, 1, 2, 2)
            [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

            df1 += df1_
            db1 += db1_
            df2 += df2_
            db2 += db2_
            dw3 += dw3_
            db3 += db3_
            dw4 += dw4_
            db4 += db4_

            cost_ += loss

        # Parameter Update
        v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size  # momentum update
        s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2  # RMSProp update
        f1 -= alpha * v1 / np.sqrt(s1 + 1e-7)  # combine momentum and RMSProp to perform update with Adam

        bv1 = beta1 * bv1 + (1 - beta1) * db1 / batch_size
        bs1 = beta2 * bs1 + (1 - beta2) * (db1 / batch_size) ** 2
        b1 -= alpha * bv1 / np.sqrt(bs1 + 1e-7)

        v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
        s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
        f2 -= alpha * v2 / np.sqrt(s2 + 1e-7)

        bv2 = beta1 * bv2 + (1 - beta1) * db2 / batch_size
        bs2 = beta2 * bs2 + (1 - beta2) * (db2 / batch_size) ** 2
        b2 -= alpha * bv2 / np.sqrt(bs2 + 1e-7)

        v3 = beta1 * v3 + (1 - beta1) * dw3 / batch_size
        s3 = beta2 * s3 + (1 - beta2) * (dw3 / batch_size) ** 2
        w3 -= alpha * v3 / np.sqrt(s3 + 1e-7)

        bv3 = beta1 * bv3 + (1 - beta1) * db3 / batch_size
        bs3 = beta2 * bs3 + (1 - beta2) * (db3 / batch_size) ** 2
        b3 -= alpha * bv3 / np.sqrt(bs3 + 1e-7)

        v4 = beta1 * v4 + (1 - beta1) * dw4 / batch_size
        s4 = beta2 * s4 + (1 - beta2) * (dw4 / batch_size) ** 2
        w4 -= alpha * v4 / np.sqrt(s4 + 1e-7)

        bv4 = beta1 * bv4 + (1 - beta1) * db4 / batch_size
        bs4 = beta2 * bs4 + (1 - beta2) * (db4 / batch_size) ** 2
        b4 -= alpha * bv4 / np.sqrt(bs4 + 1e-7)

        cost_ = cost_ / batch_size
        cost.append(cost_)

        params = [f1, f2, w3, w4, b1, b2, b3, b4]

        return params, cost