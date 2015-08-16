from time import time
from math import log

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from digits_lib import get_data, predict

X_train, X_test, y_train, y_test = get_data()

'''
t0 = time()
clf = RandomForestClassifier(n_jobs=6)
gs = GridSearchCV(clf, \
		param_grid={"max_depth": np.logspace(1.0, 5.0, num=5)})
gs.fit(X_train, y_train.label)
depth = gs.best_params_["max_depth"]
print "depth: %d" % depth
print "log(depth): %f" % log(depth, 10.0)
print "grid search: %.2fs" % (time()-t0)
'''

t0 = time()
clf = RandomForestClassifier(max_depth=1000, n_jobs=6)
clf.fit(X_train, y_train.label)
print "fitting: %.2fs" % (time()-t0)

p_train = predict(clf, X_train, y_train)
p_test = predict(clf, X_test, y_test)

print p_test.head()

print "train accuracy: %f" % (1.0*p_train.correct.sum()/p_train.shape[0])
print "test accuracy: %f" % (1.0*p_test.correct.sum()/p_test.shape[0])

write_prediction(clf, "tree_out.csv")
