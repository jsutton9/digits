import os.path
import pylearn2

import pandas as pd
import digits_lib
from sklearn.cross_validation import train_test_split

from pylearn2.utils import serial
from pylearn2.datasets import preprocessing

if __name__ == "__main__":
    data = pd.read_csv("train.csv")
    data_train, data_test = train_test_split(data, test_size=0.3)
    data_train = pd.DataFrame(data_train, columns=data.columns)
    data_test = pd.DataFrame(data_test, columns=data.columns)

    train = pd.DataFrame(index=data_train.index)
    test = pd.DataFrame(index=data_test.index)

    '''
    for i in xrange(10):
        train["is_%d" % i] = 0
        train["is_%d" % i][data_train.label==i] = 1
        test["is_%d" % i] = 0
        test["is_%d" % i][data_test.label==i] = 1
    '''
    train["label"] = data_train.label
    test["label"] = data_test.label

    for i in xrange(28*28):
        train["pixel%d" % i] = data_train["pixel%d" % i] / 255.0
        test["pixel%d" % i] = data_test["pixel%d" % i] / 255.0

    f = open("train_bin.csv", "w")
    train.to_csv(f, index=False)
    f.close()
    f = open("test_bin.csv", "w")
    test.to_csv(f, index=False)
    f.close()
