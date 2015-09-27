import random
import time

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

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

    return np.asmatrix(patch_vecs, dtype="float32")

def remove_zeros(patches):
    patch_list = []
    for i in xrange(patches.shape[0]):
        arr = [patches[i,j] for j in xrange(patches.shape[1])]
        if sum(arr) != 0:
            patch_list.append(arr)

    return np.matrix(patch_list, dtype="float32")

class Autoencoder:
    def __init__(self, input_size, output_size):
        # basic params
        rand_weights= 0.02*np.random.rand(input_size, output_size) - 0.01
        rand_bias = 0.1*np.random.rand(output_size)
        w = theano.shared(rand_weights)
        b = theano.shared(rand_bias)
        eps = theano.shared(0.0)
        dec = T.scalar("dec")
        v1s = T.matrix("v1s")
        v2s = T.matrix("v2s")
        i = theano.shared(0)
        self.eps = eps
        self.i = i
        self.weights = w

        # energy formulas
        h1s = T.tanh(T.dot(v1s, w) + b)
        h2s = T.tanh(T.dot(v2s, w) + b)
        #E = ((h2s-h1s)**2*(1-h2s**2)*(1-h1s**2)).sum()
        #E = ((h2s-h1s)**2).sum()*(2-h2s**2-h1s**2).sum()
        E = ((h2s-h1s)**2).sum()
        g_w = theano.gradient.grad(E, w)
        g_b = theano.gradient.grad(E, b)

        #training algorithm
        delta_w = g_w*eps/input_size
        delta_b = g_b*eps
        self.gradient_ascent = theano.function([v1s, v2s, dec], E, \
                updates=(\
                    (w, w+delta_w), 
                    (b, b+delta_b), 
                    (i, i+1), 
                    (eps, eps*(1-dec))
                )
        )

        # encode loop
        self.encode = theano.function([v1s], h1s)

    def train(self, data, epochs, lr, decay):
        self.i.set_value(0)
        self.eps.set_value(lr)
        for _ in xrange(epochs):
            shuffled = [[data[i,j] for j in xrange(data.shape[1])] \
                for i in xrange(data.shape[0])]
            random.shuffle(shuffled)
            v1s = np.matrix(shuffled[:len(shuffled)/2], dtype="float32")
            v2s = np.matrix(shuffled[len(shuffled)/2:], dtype="float32")
            E = self.gradient_ascent(v1s, v2s, decay)
            print "%d: avg E=%f" % (self.i.get_value(), E/len(data))

    def encode_patches(self, data):
        return self.encode(data)

if __name__=="__main__":
    t0 = time.time()
    df = pd.DataFrame.from_csv("data/train_bin.csv")
    images = df[["pixel%d"%i for i in xrange(784)]].as_matrix()
    patches = np.concatenate([patchify(image, 5, 1) for image in images])
    patches = remove_zeros(patches)
    print "prep data: %.2fs" % (time.time()-t0)

    t0 = time.time()
    encoder = Autoencoder(25, 4)
    print "build: %.2fs" % (time.time()-t0)
    t0 = time.time()
    encoder.train(patches, 5, 0.001, 0.2)
    print "train: %.2fs" % (time.time()-t0)

    shuffled = [[patches[i,j] for j in xrange(patches.shape[1])] \
            for i in xrange(patches.shape[0])]
    random.shuffle(shuffled)
    patch_sample = np.matrix(shuffled[:10], dtype="float32")
    encoding_sample = encoder.encode_patches(patch_sample)
    for i in xrange(10):
        print ""
        print patch_sample[i].reshape(5, 5)
        print encoding_sample[i]
