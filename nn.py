import pandas as pd
import time
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD

def prep_data(filename, has_y=True):
    t0 = time.time()
    data = pd.read_csv(filename)
    X = data[["pixel%d" % x for x in xrange(784)]].as_matrix()
    if has_y:
        y = data[["is_%d" % x for x in xrange(10)]].as_matrix()
    print "prep data %s: %.2fs" % (filename, time.time()-t0)
    if has_y:
        return X, y
    else:
        return X

def build_model(layer_sizes):
    t0 = time.time()
    model = Sequential()

    layer_sizes = [784] + layer_sizes
    for i in xrange(len(layer_sizes)-1):
        model.add(Dense(layer_sizes[i], layer_sizes[i+1], init="uniform"))
        model.add(Activation("tanh"))
        model.add(Dropout(0.5))
    model.add(Dense(layer_sizes[-1], 10, init="uniform"))
    model.add(Activation("softmax"))

    sgd = SGD(lr=0.1, decay=1e-6)
    model.compile(loss="mse", optimizer=sgd)
    print "prep model: %.2fs" % (time.time()-t0)
    return model

def evaluate_accuracy(model, X, y):
    t0 = time.time()
    predictions = model.predict(X, batch_size=16, verbose=2)
    predictions = [max(range(10), key=lambda x:p[x]) for p in predictions]
    correct = 0
    for i in xrange(len(predictions)):
        if y[i][predictions[i]] == 1:
            correct += 1

    print "evaluate: %.2fs" % (time.time()-t0)
    return 1.0*correct/len(predictions)

def write_predictions(model, X, filename):
    predictions = model.predict(X, batch_size=16, verbose=2)
    predictions = [max(range(10), key=lambda x:p[x]) for p in predictions]
    f = open(filename, "w")
    f.write("ImageId,Label\n")
    for i in xrange(len(predictions)):
        f.write("%d,%d\n" % (i+1,predictions[i]))
    f.close()

def timed_run(model, time_limit):
    t_start = time.time()
    times = []
    accuracies = []
    while time.time() < t_start + time_limit*60:
        t0 = time.time()
        model.fit(X_train, y_train, nb_epoch=5, batch_size=16, verbose=2)
        t = time.time()-t0
        print "fit: %d:%02d" % (int(t/60), int(t%60))

        accuracy = evaluate_accuracy(model, X_test, y_test)
        times.append((time.time()-t_start)/60)
        accuracies.append(accuracy)
        print "accuracy: %.3f%%" % (100*accuracy)

    return times, accuracies

def patchify(image, size, overlap):
    padding = (overlap-28) % (size-overlap)
    image_list = image.tolist()
    image_matrix = [image_list[28*i:28*(i+1)] for i in xrange(28)]
    for row in image_matrix:
        row += row[:padding]
    image_matrix += image_matrix[:padding]

    patch_vecs = []
    for y1 in xrange(0, 28-overlap, size-overlap):
        y2 = y1 + size
        rows = image_matrix[y1:y2]
        for x1 in xrange(0, 28-overlap, size-overlap):
            x2 = x1 + size
            patch_matrix = [row[x1:x2] for row in rows]
            patch_vecs.append(reduce(lambda a,b:a+b, patch_matrix))

    return patch_vecs

X_train, y_train = prep_data("data/train_bin.csv")
X_test, y_test = prep_data("data/test_bin.csv")
X_target = prep_data("data/target_adj.csv", has_y=False)

'''
model = build_model([1000, 1600, 400])
#model.load_weights("data/1000-1600-400-0hrs.hdf5")
times, accuracies = timed_run(model, 60)
model.save_weights("data/1000-1600-400-1hrs.hdf5", overwrite=True)
plt.plot(times, accuracies, label="[1000, 1600, 400]")
'''

model = build_model([500, 200])
model.load_weights("data/500-200-8hrs.hdf5")
write_predictions(model, X_target, "data/500-200-8hrs_out.csv")

#plt.legend(loc="lower right")
#plt.show()
