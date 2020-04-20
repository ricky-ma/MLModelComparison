# MNIST From Scratch
Comparison of machine learning models built from scratch, trained and tested on the MNIST dataset. Models and techniques implemented include K-nearest neighbours, linear/logistical multi-class regression, standard/stochastic gradient descent, principal component analyses, support vector machines, neural networks, and convolution neural networks.

## Introduction

## Methods

### K-Nearest Neighbours (KNN)
A hyperparameter of k=3 resulted in a test error of $3.19\%$. Being non-parametric, a KNN implementation with such a large dataset resulted in very long prediction times ($>$ 20 mins for k=1 and k=2) due to the calculation, and especially sorting, of Euclidean distances. While doing these computations locally, both memory and disk space frequently maxed out. Nevertheless, the test error is rather low, even without other preprocessing techniques.

### Logistic Regression with Principal Component Analysis (PCA)

### Support Vector Machines (SVM)

### Multi Layer Perceptron (MLP)
A hidden layer size of 500 was used, which took about 3 hours to train. An alpha of 0.001 performed best, with both 0.0005 and 0.005 increasing test error. A batch size of 2500 significantly helped to reduce overfitting. An epoch of 250 worked well. More epochs may yield better test errors, but training time is already significantly lengthy. SGD was used to decrease training time for hyperparameter tuning, while traditional gradient descent was used to achieve the best model.

### Convolutional Neural Network (CNN)

## Results
| Model               | Test Error (%)|
| -------------       |:-------------:|
| KNN                 | 3.19          |
| softmax regression  | 19.97         |
| SVM                 |               |
| MLP                 | 1.57          |
| CNN                 |               |

## Discussion
