import numpy as np
import math
import pickle
import time
from numba import *
from sklearn.cluster import KMeans

from matrix_compute import get_gxyz
from filter_func import get_normalized_gaussian
from kmeans_vector import KMeans_Vector
from preprocessing import get_array_data

import filter_constant as C

@njit
def get_invarient_set(l1, l2, l3):
    l1 = math.sqrt(l1)
    l2 = math.sqrt(l2)
    l3 = math.sqrt(l3)
    trace = l1 + l2 + l3
    mean_la = trace / 3
    fa = ((l1 - mean_la)**2 + (l2 - mean_la)**2 + (l3 - mean_la)**2)/ (l1**2 + l2**2 + l3**2)
    mode = (-l1-l2+2*l3)*(2*l1-l2-l3)*(-l1+2*l2-l3)/2/pow(l1**2+l2**2+l3**2-l1*l2-l1*l3-l2*l3, 1.5)
    return trace, fa, mode

@njit
def get_features(patchX, patchY, patchZ, weight):
    G = np.vstack((patchX.ravel(), patchY.ravel(), patchZ.ravel())).T
    x = G.T @ (weight * G)
    w, v = np.linalg.eig(x)

    index = w.argsort()[::-1]
    [l1, l2, l3] = w[index]
    v = v[:, index]

    v1 = v[:, 0]
    index1 = np.abs(v1).argsort()[::-1]
    v1 = v1[index1]
    sign = np.sign(v1)
    v1 = np.abs(v1)

    invarient_set = get_invarient_set(l1, l2, l3)

    trace = math.sqrt(invarient_set[0])
    fa = math.sqrt(invarient_set[1])
    mode = math.sqrt(invarient_set[2]+1.0001)

    return v1[0], v1[1], v1[2], trace, fa, mode, index1, sign


@njit
def make_point_space(im_LR, im_GX, im_GY, im_GZ, patchNumber, w, point_space, MAX_POINTS):
    H, W, D = im_GX.shape

    for i1 in range(C.PATCH_HALF, H - C.PATCH_HALF):
        for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
            for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):

                if im_LR[i1, j1, k1] == 0:
                    continue

                patchX, patchY, patchZ = get_gxyz(im_GX, im_GY, im_GZ, i1, j1, k1)

                point_space[patchNumber] = np.array(get_features(patchX, patchY, patchZ, w)[:-2])
                patchNumber += 1

    return point_space, patchNumber


def init_buckets(Q_TOTAL):
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS


def k_means_modeling(quantization):
    kmeans_angle = KMeans_Vector(n_clusters=C.Q_ANGLE, verbose=True, max_iter=30, n_init=1)
    kmeans_angle.fit(quantization[:, :3])

    kmeans_tensor = KMeans(n_clusters=C.Q_TENSOR, verbose=True, max_iter=30, n_init=1)
    kmeans_tensor.fit(quantization[:, 3:])

    return kmeans_angle, kmeans_tensor


def make_kmeans_model(file_list):
    G_WEIGHT = get_normalized_gaussian()

    MAX_POINTS = 15000000
    patchNumber = 0
    point_space = np.zeros((MAX_POINTS, 6))

    for file_idx, file in enumerate(file_list):
        print('\r', end='')
        print('' * 60, end='')
        print('\r Making Point Space: '+ file.split('\\')[-1] + str(MAX_POINTS) + ' patches (' + str(100*patchNumber/MAX_POINTS) + '%)')

        im_HR, im_LR = get_array_data(file, training=True)
        im_GX, im_GY, im_GZ = np.gradient(im_LR)

        point_space, patchNumber = make_point_space(im_LR, im_GX, im_GY, im_GZ, patchNumber, G_WEIGHT, point_space, MAX_POINTS)
        if patchNumber > MAX_POINTS / 2:
            break

    point_space = point_space[0:patchNumber, :]
    # point_square = point_space[:, 3:] * point_space[:, 3:]
    # std = np.std(point_square, axis=0)
    # point_space[:, 3:] = np.sqrt(point_square * (1/std)[None, :])

    start = time.time()
    print('start clustering')
    kmeans = k_means_modeling(point_space)
    print(time.time() - start)

    with open('./arrays/space_{}x_{}.km'.format(C.R, C.Q_TOTAL), 'wb') as p:
        pickle.dump(kmeans, p)

    return kmeans


def load_kmeans_model():
    with open('./arrays/space_{}x_{}.km'.format(C.R, C.Q_TOTAL), "rb") as p:
        kmeans = pickle.load(p)
    return kmeans