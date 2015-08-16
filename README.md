Digits
======

This code is for my entry in Kaggle's <a href="https://www.kaggle.com/c/digit-recognizer/data">Digit Recognizer competition</a>. This is based on the MNIST handwritten digit data set, which consists of 28 by 28 pixel grayscale images of handwritten digits. The competition supplies a labeled training set (train.csv) and an unlabeled test set (target.csv), where the goal is to identify the digits in the test set.

Using scikit-learn, I have tried logistic regression, support vector machine, and decision tree models. Of these, the support vector machine has been the most effective, with around 93 percent prediction accuracy. Now, I am working on using pylearn2 to implement a restricted Boltzmann machine.
