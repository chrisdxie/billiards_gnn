"""
This script comes from the RTRBM code by Ilya Sutskever from
http://www.cs.utoronto.ca/~ilya/code/2008/RTRBM.tar

Code has been adapted from: https://github.com/stelzner/Visual-Interaction-Networks/blob/master/create_billards_data.py
"""

import os
import argparse

from numpy import *
from scipy import *
import scipy.io

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import visualize_billiards as vis

shape_std = shape


def shape(A):
    if isinstance(A, ndarray):
        return shape_std(A)
    else:
        return A.shape()


size_std = size


def size(A):
    if isinstance(A, ndarray):
        return size_std(A)
    else:
        return A.size()


det = linalg.det


def new_speeds(m1, m2, v1, v2):
    new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
    new_v1 = new_v2 + (v2 - v1)
    return new_v1, new_v2


def norm(x): return sqrt((x ** 2).sum())


def sigmoid(x): return 1. / (1. + exp(-x))



# size of bounding box: SIZE X SIZE.

def bounce_n(T=128, n=2, r=None, m=None):
    # n: number of balls
    # T: length of sequence
    # r: radius
    # m: mass

    if r is None:
        r = array([1.2] * n)
    if m is None:
        m = array([1] * n)
    # r is to be rather small.
    X = zeros((T, n, 2), dtype='float') # Position...?
    y = zeros((T, n, 2), dtype='float') # velocity...?
    v = randn(n, 2)
    v = v / norm(v) * .5 # initial velocity

    # Initial location. Make sure they don't hit bbox and don't overlap
    good_config = False
    while not good_config:
        x = 2 + rand(n, 2) * (SIZE-2) # Shape: [n x 2]
        good_config = True

        # Check box boundary conditions
        for i in range(n):
            for z in range(2):
                if x[i][z] - r[i] < 0:
                    good_config = False
                if x[i][z] + r[i] > SIZE:
                    good_config = False

        # Check overlap
        for i in range(n):
            for j in range(i):
                if norm(x[i] - x[j]) < r[i] + r[j]:
                    good_config = False

    eps = .5
    for t in range(T):
        # for how long do we show small simulation

        v_prev = copy(v)

        for i in range(n):
            X[t, i] = x[i]
            y[t, i] = v[i]

        for mu in range(int(1 / eps)):

            for i in range(n):
                x[i] += eps * v[i] # Euler integration

            # Check for boundary conditions (bouncing off of the wall)
            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0:
                        v[i][z] = abs(v[i][z])  # want positive
                    if x[i][z] + r[i] > SIZE:
                        v[i][z] = -abs(v[i][z])  # want negative

            # Check for bouncing off of each other
            for i in range(n):
                for j in range(i):
                    if norm(x[i] - x[j]) < r[i] + r[j]:

                        w = x[i] - x[j]
                        w = w / norm(w)

                        v_i = dot(w.transpose(), v[i])
                        v_j = dot(w.transpose(), v[j])

                        new_v_i, new_v_j = new_speeds(m[i], m[j], v_i, v_j)

                        v[i] += w * (new_v_i - v_i)
                        v[j] += w * (new_v_j - v_j)

    return X, y


def ar(x, y, z):
    return z / 2 + arange(x, y, z, dtype='float')


def draw_image(X, res, r=None):
    T, n = shape(X)[0:2]
    if r is None:
        r = array([1.2] * n)

    A = zeros((T, res, res, 3), dtype='float')

    [I, J] = meshgrid(ar(0, 1, 1. / res) * SIZE, ar(0, 1, 1. / res) * SIZE)

    for t in range(T):
        for i in range(n):
            A[t, :, :, i] += exp(-(((I - X[t, i, 0]) ** 2 +
                                    (J - X[t, i, 1]) ** 2) /
                                   (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1
    return A


def bounce_vec(res, n=2, T=128, r=None, m=None):
    # n: number of balls
    # T: length of sequence
    # r: radius
    # m: mass

    if r is None:
        r = array([1.2] * n)
    x, y = bounce_n(T, n, r, m)
    y = concatenate((x, y), axis=2)
    return y


def show_sample(V):
    T = V.shape[0]
    for t in range(T):
        plt.imshow(V[t])
        # Save it
        fname = logdir + '/' + str(t) + '.png'
        plt.savefig(fname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('N_train', help="Number of sequences") # default=1000
    parser.add_argument('N_test', help="Number of sequences") # default=200
    parser.add_argument('n', help="Number of objects") # default=3
    parser.add_argument('size', help="Size of bounding box") # default=10
    parser.add_argument('T', help="Length of each sequence") # default=100
    parser.add_argument('res', help="Resolution") # default=32
    parser.add_argument('logdir', help="Directory to save images")
    args = parser.parse_args()

    N_train = int(args.N_train) # Number of sequences
    N_test = int(args.N_test) # Number of sequences
    T = int(args.T) # Length of each sequence
    res = int(args.res) # Image resolution
    n = int(args.n) # Number of objects

    global SIZE
    SIZE = int(args.size)
    global logdir
    logdir = args.logdir

    # Generate Test
    dat_y = empty((N_train, T, n, 4), dtype=float)
    for i in range(N_train):
        dat_y[i] = bounce_vec(res=res, n=n, T=T)
        print('training example {} / {}'.format(i, N_train))
    data = dict()
    data['y'] = dat_y
    scipy.io.savemat(os.path.join(logdir,'billiards_balls_training_data.mat'), data)

    dat_y = empty((N_test, T, n, 4), dtype=float)
    for i in range(N_test):
        dat_y[i] = bounce_vec(res=res, n=n, T=T)
        print('test example {} / {}'.format(i, N_test))
    data = dict()
    data['y'] = dat_y
    scipy.io.savemat(os.path.join(logdir,'billiards_balls_testing_data.mat'), data)

    # show one video
    vis.animate(dat_y[0], logdir, 'ex', window_size=SIZE)
