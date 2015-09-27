import time
import random

import numpy as np
import pandas as pd
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

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

class RBM:
    def __init__(self, input_size, output_size):
        # RANDOM INITIAL STATE
        rand_weights = 0.02*np.random.rand(input_size, output_size) - 0.01
        rand_v_bias = 0.1*np.random.rand(input_size) - 0.05
        rand_h_bias = 0.1*np.random.rand(output_size) - 0.05
        w = theano.shared(rand_weights)
        a = theano.shared(rand_v_bias)
        b = theano.shared(rand_h_bias)

        # TRAINING ALGORITHM
        # inputs
        vs_raw = T.matrix("vs_raw")

        # randomness for activation
        rng = RandomStreams()
        vs_thresh = rng.uniform((vs_raw.shape[0], input_size))
        hs_thresh = rng.uniform((vs_raw.shape[0], output_size))
        vs_p_thresh = rng.uniform((vs_raw.shape[0], input_size))
        hs_p_thresh = rng.uniform((vs_raw.shape[0], output_size))

        # make inputs binary
        vs = vs_raw > vs_thresh

        # sample hidden activations
        hs_in = b + T.dot(vs, w)
        hs_prob = 1/(1 + T.exp(-hs_in))
        hs = hs_prob > hs_thresh

        # reconstruct inputs (Gibbs sampling)
        vs_p_in = a + T.dot(hs, w.T)
        vs_p_prob = 1/(1 + T.exp(-vs_p_in))
        vs_p = vs_p_prob > vs_p_thresh

        # resample hidden activations (Gibbs sampling)
        hs_p_in = b + T.dot(vs, w)
        hs_p_prob = 1/(1 + T.exp(-hs_p_in))
        hs_p = hs_p_prob > hs_p_thresh

        # contrastive divergence
        eps = theano.shared(0.0)
        decay = T.scalar("decay")
        pos_grad = T.tensordot(vs, hs, axes=[0, 0])
        neg_grad = T.tensordot(vs_p, hs_p, axes=[0, 0])
        delta_w = eps*(pos_grad-neg_grad)/(input_size*output_size)
        delta_a = eps*(vs-vs_p).sum(axis=0)/input_size
        delta_b = eps*(hs-hs_p).sum(axis=0)/output_size
        cd = theano.function([vs_raw, decay], [], updates=(\
                (w, w+delta_w), (a, a+delta_a),
                (b, b+delta_b), (eps, eps*(1-decay))
        ))

        # store algorithm
        self.eps = eps
        self.cd = cd

        # EVALUATION ALGORITHM
        # all possible hidden configurations
        all_hs = []
        for x in xrange(2**output_size):
            all_hs.append([(x>>i)&1 for i in xrange(output_size)])
        all_hs = np.array(all_hs, dtype="int8")

        # get v_i probabilities for each possible hidden configuration
        v_ins_all_hs = a + T.dot(all_hs, w.T)
        v_probs_all_hs = 1/(1 + T.exp(-v_ins_all_hs))
        get_v_prob_matrix = theano.function([], v_probs_all_hs)

        # make input binary
        v_thresh = rng.uniform((input_size,))
        v_raw = T.vector("v_raw")
        v = v_raw > v_thresh

        # probability of given input vector over hidden configurations
        v_prob_matrix = T.matrix("v_prob_matrix")
        v_probs = v_prob_matrix*v + (1-v_prob_matrix)*(1-v)
        #v_prob = T.sum(T.prod(v_probs, axis=1))/v_probs.shape[0]
        v_probs_prod = T.prod(v_probs, axis=1)
        v_prob = T.sum(v_probs_prod)/v_probs.shape[0]
        #get_v_prob = theano.function([v_prob_matrix, v_raw], v_prob)
        get_v_prob = theano.function([v_prob_matrix, v_raw], \
                [v_prob, v_probs_prod])
        # If I remove v_probs_prod from the return values of get_v_prob, 
        # I get a bug where it takes a sum instead of a product.
        # I posted an issue ticket on github - #3451

        # get expected log probability of data
        vs_prob = T.vector("vs_prob")
        mean_log_prob = T.sum(T.log(vs_prob))/vs_prob.shape[0]
        get_mean_log_prob = theano.function([vs_prob], mean_log_prob)

        # store algorithms
        self.get_v_prob_matrix = get_v_prob_matrix
        self.get_v_prob = get_v_prob
        self.get_mean_log_prob = get_mean_log_prob

    def train(self, data, lr, decay, batch_size, epochs):
        self.eps.set_value(lr)
        for i in xrange(epochs):
            t0 = time.time()
            shuffled_data = [row for row in np.asarray(data)]
            random.shuffle(shuffled_data)
            shuffled_data = np.matrix(shuffled_data)
            for j in xrange(shuffled_data.shape[0]/batch_size):
                batch = shuffled_data[j*batch_size:(j+1)*batch_size]
                self.cd(batch, decay)
            fitness = self.get_log_prob(shuffled_data[:1000])
            print "  epoch %d (%.2fs); mean log prob: %f" \
                    % (i, time.time()-t0, fitness)

    def get_log_prob(self, data):
        v_probs_all_hs = np.matrix(self.get_v_prob_matrix(), \
                dtype="float32")
        vs_prob_values = [self.get_v_prob(v_probs_all_hs, v_raw)[0] \
                for v_raw in np.asarray(data)]
        print vs_prob_values[:10]
        '''
        vs_prob_values = []
        for v_raw in np.asarray(data):
            v_prob, _ = self.get_v_prob(v_probs_all_hs, v_raw)
            #print "v_probs"
            #print v_probs
            #print "prod"
            #print test_a
            #print "sum"
            #print test_b
            #print "v_prob"
            #print v_prob
            #print ""
            vs_prob_values.append(v_prob)
            '''
        print vs_prob_values[:10]
        vs_prob_values = np.array(vs_prob_values, dtype="float32")
        return self.get_mean_log_prob(vs_prob_values)

if __name__=="__main__":
    t0 = time.time()
    df = pd.DataFrame.from_csv("data/train_bin.csv")
    images = df[["pixel%d"%i for i in xrange(784)]].as_matrix()
    patches = np.concatenate([patchify(image, 5, 1) for image in images])
    patches = patches[:500000]
    patches = remove_zeros(patches)
    print "prep data: %.2fs" % (time.time()-t0)

    t0 = time.time()
    rbm = RBM(25, 4)
    print "build: %.2fs" % (time.time()-t0)
    t0 = time.time()
    rbm.train(patches, 0.001, 0.1, patches.shape[0], 100)
    print "train: %.2fs" % (time.time()-t0)
