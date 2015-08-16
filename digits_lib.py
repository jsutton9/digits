import warnings
from time import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split

warnings.filterwarnings("ignore", message=".*deprecated.*")

def prepare_X(data):
    X = data[["pixel%d" % x for x in xrange(784)]]
    X /= 255.0
    mean = X.mean()
    X -= mean
    return X

def get_data():
    t0 = time()
    data = pd.read_csv("train.csv")
    data_train, data_test = train_test_split(data, test_size=0.3)
    data_train = pd.DataFrame(data_train, columns=data.columns)
    data_test = pd.DataFrame(data_test, columns=data.columns)
    X_train = prepare_X(data_train)
    X_test = prepare_X(data_test)
    y_train = data_train[["label"]]
    y_test = data_test[["label"]]
    print "prepare data: %.2fs" % (time()-t0)
    return X_train, X_test, y_train, y_test

def predict(clf, X, y):
    t0 = time()
    p = pd.DataFrame()
    p["label"] = y.label
    p["pred"] = clf.predict(X)
    p["correct"] = 0
    p.correct[p.pred==p.label] = 1
    print "verify: %.2fs" % (time()-t0)
    return p

def display_image(data):
    img = []
    for p in data:
        if p >= 0:
            img.append([p, p, p])
        else:
            img.append([-p, 0, 0])
    img = np.reshape(img, (28, 28, 3))
    plt.imshow(img)
    plt.show()

def write_prediction(clf, f_name="out.csv"):
    t0 = time()
    X = prepare_X(pd.read_csv("target.csv"))
    pred = clf.predict(X)

    f = open(f_name)
    for x in pred:
        f.write("%d\n" % x)
    f.close()
    print "predict: %.2fs" % (time()-t0)
