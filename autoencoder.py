import random

import numpy as np
import pandas as pd
import theano
import theano.tensor as T

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

class Autoencoder:
    def __init__(self, input_size, output_size):
        # basic params
        rands = 2*np.random.rand(output_size, input_size) - 1
        w = theano.shared(rands)
        eps = T.scalar("eps")
        dec = T.scalar("dec")
        v1 = T.vector("v1")
        v2 = T.vector("v2")
        i = theano.shared(0)
        self.weights = w

        # energy formulas
        h1 = T.tanh(T.dot(w, v1))
        h2 = T.tanh(T.dot(w, v2))
        delta_h = h2 - h1
        E = T.dot(delta_h, delta_h)
        g_w = T.grad(E, w)

        # training algorithm
        delta_w = g_w*eps/(1.0+i*dec)
        gradient_ascent = theano.function([v1, v2, eps, dec], E, \
                updates=((w, w+delta_w), (i, i+1)))
        v1s = T.matrix("v1s")
        v2s = T.matrix("v2s")

        # training loop
        results, updates = theano.scan(fn=gradient_ascent,\
                sequences=[v1s,v2s],
                non_sequences=[eps, dec])
        E_final = results[-1]
        self.run_epoch = theano.function([v1s, v2s, eps, dec], \
                E_final, updates=updates)

        # encode loop
        encode = theano.function([v1], h1)
        results, updates = theano.scan(fn=encode,\
                sequences=[v1s])
        self.encode_loop = theano.function([v1s], results, \
                updates=updates)

    def train(self, data, epochs, lr, decay):
        for _ in xrange(epochs):
            shuffled = random.shuffle(data[:])
            mid = len(shuffled)/2
            E_final_val = self.run_epoch(shuffled[:mid], \
                    shuffled[mid:], lr, decay)
            print "%d: E=%f" % (i.value, E_final_val)

    def encode_patches(self, data):
        return self.encode_loop(data)

if __name__=="__main__":
    df = pd.DataFrame.from_csv("data/train_bin.csv")
    images = df[["pixel%d"%i for i in xrange(784)]].as_matrix()
    patches = patchify(images, 5, 1)
    #encoder = Autoencoder(25, 4)
    encoder = Autoencoder(3, 2)
    #encoder.train(patches, 5, 0.1, 1e-4)
