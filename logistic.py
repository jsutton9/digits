from time import time

from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def to_img(data, mean):
    adj = 1 - data - mean
    img = []
    for p in adj:
        if p >= 0:
            img.append([p, p, p])
        else:
            img.append([-p, 0, 0])
    img = np.reshape(img, (28, 28, 3))
    return img

def prepare_X(data):
    X = data[["pixel%d" % x for x in xrange(784)]]
    X /= 255.0
    mean = X.mean()
    X -= mean
    return X

def make_guesses(p, y):
    guesses = y[["label"]]
    guesses["guess"] = [max(range(10), key=lambda x:p.loc[i]["p_%d" % x]) \
            for i in xrange(guesses.shape[0])]
    guesses["correct"] = 0
    guesses.correct[guesses.guess==guesses.label] = 1
    return guesses

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

'''
t0 = time()
pca = RandomizedPCA(n_components=20)
pca.fit(X_train)
X_train_proj = pd.DataFrame(pca.transform(X_train), index=X_train.index)
print "train pca, transform: %.2fs" % (time()-t0)
'''

t0 = time()
model = LogisticRegression()
#model.fit(X_train_proj, y_train.label)
model.fit(X_train, y_train.label)
'''
model = LogisticRegression()
gs = GridSearchCV(model, param_grid={"C": np.logspace(-4, 4, num=10)})
gs.fit(X_train, y_train.label)
C = gs.best_params_["C"]
model = LogisticRegression(C=C)
model.fit(X_train, y_train.label)
'''
print "train fitting: %.2fs" % (time()-t0)

'''
t0 = time()
X_test_proj = pd.DataFrame(pca.transform(X_test), index=X_test.index)
print "test transform: %.2fs" % (time()-t0)
'''

t0 = time()
#p_test = pd.DataFrame(model.predict_proba(X_test_proj), columns=["p_%d" % i for i in xrange(10)], index=X_test_proj.index)
p_test = pd.DataFrame(model.predict_proba(X_test), columns=["p_%d" % i for i in xrange(10)], index=X_test.index)
print "test prediction: %.2fs" % (time()-t0)

t0 = time()
guesses_test = make_guesses(p_test, y_test)
print "guesses_test: %.2fs" % (time()-t0)

print "test accuracy: %f" % (1.0*guesses_test.correct.sum() / guesses_test.shape[0])

write_prediction(clf, "logistic_out.csv")
