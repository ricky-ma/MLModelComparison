# MNIST From Scratch
Comparison of machine learning models built from scratch, trained and tested on the MNIST dataset.

## Introduction
The MNIST dataset consists of 70,000 examples of handwritten digits, split into 60,000 training examples and 10,000 test examples. Classification on this dataset involves labeling each handwritten digit as an integer from 0 to 9. Numerous machine learning models exist for multi-class classification problems like this. This project covers 5 different approaches, from linear regression to convolutional neural nets, using various optimization, regularization, and  hyperparameter tuning techniques.

## Methods

### K-Nearest Neighbours (KNN)
A hyperparameter of k=3 resulted in a test error of 3.19%. Being non-parametric, a KNN implementation with such a large dataset resulted in relatively long prediction times (> 20 mins for k=1 and k=2) due to the calculation, and especially sorting, of Euclidean distances. While doing these computations locally, both memory and disk space frequently maxed out. Nevertheless, the test error is rather low, even without other preprocessing techniques.

### Linear Regression
Softmax loss was used to implement a multi-class logistic regression. L2 regularization resulted in the best accuracy out of L0, L1, and L2 regularization techniques. A bias variable was also added to improve accuracy. Optimization with SGD resulted in a test error of 19.97 with a batch size of 5000 and alpha of 0.1. Only 2 epochs were used, as loss stopped decreasing past 2 epochs.

### Support Vector Machine (SVM)
A support vector machine using L2 regularization and optimized with SGD was implemented to achieve a test error of 9.13\%. Hinge loss was calculated naively in python without the use of vectors. SGD was implemented with step decay such that alpha was halved every 5 epochs. This greatly reduced training time and improved test accuracy, as alpha started out at 0.1 and ended at 0.0125. The model was trained for a total of 20 epochs, with a batch sizes of 2500. Loss plateaued around epoch 18. 

### Multi-Layer Perceptron (MLP)
A hidden layer size of 500 was used, which took about 3 hours to train. An alpha of 0.001 performed best, with both 0.0005 and 0.005 increasing test error. A batch size of 2500 significantly helped to reduce overfitting. An epoch of 250 worked well. More epochs may yield better test errors, but training time is already significantly lengthy. SGD was used to decrease training time for hyperparameter tuning, while traditional gradient descent was used to achieve the best model.

### Convolutional Neural Network (CNN)
Adam optimization, a combination of momentum and RMSProp, was used to optimize the CNN. Each learning rate alpha depends on the gradient, and is adjusted with every learning step. ReLU activation was used, with 4 epochs and batch sizes of 64.  Weights were randomly initialized to avoid 0 gradients. Adjusting the number of epochs and batch sizes may result in better accuracy, but training is lengthy and costly.

## Results
| Model               | Test Error (%)|
| -------------       |:-------------:|
| KNN                 | 3.19          |
| linear regression   | 19.97         |
| SVM                 | 9.13          |
| MLP                 | 1.57          |
| CNN                 | 1.63          |

## Discussion
