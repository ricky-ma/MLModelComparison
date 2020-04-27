# MNIST From Scratch
Comparison of machine learning models built from scratch, trained and tested on the MNIST dataset.

## Introduction
The MNIST dataset consists of 70,000 examples of handwritten digits, split into 60,000 training examples and 10,000 test examples. Classification on this dataset involves labeling each handwritten digit as an integer from 0 to 9. Numerous machine learning models exist for multi-class classification problems like this. This project covers 5 different approaches, from linear regression to convolutional neural nets, using various optimization, regularization, and  hyperparameter tuning techniques.

## Methods

### K-Nearest Neighbours (KNN)
A hyperparameter of k=3 resulted in a test error of 3.19%. Being non-parametric, a KNN implementation with such a large dataset resulted in relatively long prediction times (> 20 mins for k=1 and k=2) due to the calculation, and especially sorting, of Euclidean distances. While doing these computations locally, both memory and disk space frequently maxed out. Nevertheless, the test error is rather low, even without other preprocessing techniques.

### Linear Regression
Softmax loss was used for multiclass logistical linear regression. L2 regularization resulted in the best accuracy out of L0, L1, and L2 regularization techniques. A bias variable was also added to improve accuracy. Optimization with SGD resulted in a test error of 19.97 with a batch size of 5000 and alpha of 0.1. Smaller batch sizes and lower alphas resulted in very long training times. Only 2 epochs were used, as loss stopped decreasing past 2 epochs.

### Support Vector Machine (SVM)
A support vector machine using L2 regularization and optimized with SGD was implemented to achieve a test error of 9.13%. Hinge loss was calculated naively in python without the use of vectors. SGD was implemented with step decay such that alpha was halved every 5 epochs. This greatly reduced training time and improved test accuracy, as alpha started out at 0.1 and ended at 0.0125. The model was trained for a total of 20 epochs, with a batch sizes of 2500. Loss plateaued around epoch 18. 

### Multi-Layer Perceptron (MLP)
A hidden layer size of 500 was used, which took about 3 hours to train. An alpha of 0.001 performed best, with both 0.0005 and 0.005 increasing test error. A batch size of 2500 helped to reduce overfitting, as smaller batch sizes resulted in higher test error. 250 epochs worked well. More epochs may yield better test errors, but training time is already significantly lengthy. SGD was used to decrease training time for hyperparameter tuning, while traditional gradient descent was used to achieve the best accuracy.

### Convolutional Neural Network (CNN)
Adam optimization, a combination of momentum and RMSProp, was used to optimize the CNN. Each learning rate alpha depends on the gradient, and is adjusted with every learning step. ReLU activation was used, with 4 epochs and batch sizes of 64. The different between 2 and 4 epochs were not significant. Batch sizes of 64 clearly outperformed batch sizes of 32, but result in longer training times. Weights were randomly initialized to avoid 0 gradients. Increasing the number of epochs and batch sizes further may result in better accuracy, but training is lengthy and costly.

## Results
| Model               | Test Error (%)| MNIST Lowest Test Error (%) |
| -------------       |:-------------:| :------------------------:|
| KNN                 | 3.19          | 0.52                      |
| linear regression   | 19.97         | 7.6                       |
| SVM                 | 9.13          | 0.56                      |
| MLP                 | 1.57          | 0.35                      |
| CNN                 | 1.63          | 0.23                      |

## Discussion
The results from the various models implemented are compared with the best classification results 
from various studies, which can be found at http://yann.lecun.com/exdb/mnist/. 

In the best KNN approach by Keysers et. al (2007), they used a preprocessing technique called P2DHMDM, a type of nonlinear deformation. This shifts the edges of the handwritten digits, creating augmented training data, which significantly improves test accuracy. My KNN implementation contains no data preprocessing, and uses a naive Euclidean distance computation. Nevertheless, I was rather impressed by the accuracy of this basic model.

The linear classifier implemented by Yann LeCun (1998) also preprocesses the training data by deskewing. Additionally, instead of a traditional linear classifier, the best implementation uses a pairwise approach to train each unit in the layer to recognize one class. This significantly outperforms traditional regression approaches.

The best SVM approach, by DeCoste and Scholkopf (2002), uses a 9-degree polynomial kernel, along with deskewing preprocessing. My approach, however, relies on the naive optimization of hinge loss without the use of kernels. Additionally, they use virtual support vectors instead of classic support vectors. Virtual support vectors are generated from prior knowledge using traditional support vectors found in the first training run. This may explain the drastic difference in test errors. 

The best neural net approach for the MNIST classification problem was implemented by Ciresan et. al (2010). In order to achieve such a low error rate, they used 6 hidden layers with many neurons per layer, numerous deformed training images, and GPU computation. Without a GPU to speed up training, I was only able to implement a single hidden layer with 500 neurons, which took 3 hours to train. As the model developed by Ciresan et. al is fantastically simple, any ML engineer with adequate graphics cards should be able to achieve similar results.

The lowest test error on the MNIST classification problem by far was achieved by Ciresan et. al in 2012 using a committee of 35 convolutional neural networks, with width normalization preprocessing. Their algorithm imposes a winner-takes-all strategy for selecting neurons to result in several deep neural columns that are experts on inputs preprocessed in different ways. They also use GPUs to speed up training. My implementation uses a single CNN with traditional forwards and backwards convolutions, pooling, and ReLU activation. Additionally, without a GPU, hyperparameter tuning was inconvenient.

