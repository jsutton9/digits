Digits
======

This code is for my entry in Kaggle's <a href="https://www.kaggle.com/c/digit-recognizer">Digit Recognizer competition</a>. This is based on the MNIST handwritten digit data set, which consists of 28 by 28 pixel grayscale images of handwritten digits. The competition supplies a labeled training set (train.csv) and an unlabeled test set (target.csv), where the goal is to identify the digits in the test set.

Using scikit-learn, I have tried logistic regression, support vector machine, and decision tree models. Of these, the support vector machine has been the most effective, with around 93 percent prediction accuracy. Using Keras I have implemented neural nets, with which I have so far acheived accuracies up to about 97.2 percent. Next, I plan to train an unsupervised neural net on patches of the images in order to get a more efficient encoding to allow for faster training of neural nets.
