from time import time
from sklearn.svm import SVC
from digits_lib import get_data, predict, display_image, write_prediction

X_train, X_test, y_train, y_test = get_data()

t0 = time()
clf = SVC()
clf.fit(X_train, y_train.label)
print "fit: %.2fs" % (time()-t0)

#p_train = predict(clf, X_train, y_train)
#p_test = predict(clf, X_test, y_test)

#print "train accuracy: %f" % (1.0*p_train.correct.sum()/p_train.shape[0])
#print "test accuracy: %f" % (1.0*p_test.correct.sum()/p_test.shape[0])

write_prediction(clf, "svm_out.csv")
