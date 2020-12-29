import numpy as np
import math
import pickle
import time
import random
from numba import *
from sklearn.cluster import KMeans

from matrix_compute import get_gxyz
from filter_func import get_normalized_gaussian
from kmeans_tensor import KMeans_Tensor
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
def get_features(patchX, patchY, patchZ, weight, std):
    G = np.vstack((patchX.ravel(), patchY.ravel(), patchZ.ravel())).T
    x = G.T @ (weight * G)
    w, v = np.linalg.eig(x)

    index = w.argsort()[::-1]
    [l1, l2, l3] = w[index]
    v = v[:, index]

    # v1 = v[:, 0]
    # index1 = np.abs(v1).argsort()[::-1]
    # v1 = v1[index1]
    # sign = np.sign(v1)
    # v1 = np.abs(v1)

    invarient_set = get_invarient_set(l1, l2, l3)

    trace = math.sqrt(invarient_set[0] / std[0])
    fa = math.sqrt(invarient_set[1] / std[1])
    mode = math.sqrt((invarient_set[2]+1.0001) / std[2])

    #return v1[0], v1[1], v1[2], trace, fa, mode, index1, sign
    return trace, fa, mode, v


@njit
def make_point_space(im_LR, im_GX, im_GY, im_GZ, patchNumber, w, point_invarient, point_rotate, MAX_POINTS):
    H, W, D = im_GX.shape

    for i1 in range(C.PATCH_HALF, H - C.PATCH_HALF):
        for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
            for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):

                if im_LR[i1, j1, k1] == 0:
                    continue

                patchX, patchY, patchZ = get_gxyz(im_GX, im_GY, im_GZ, i1, j1, k1)
                features = get_features(patchX, patchY, patchZ, w, np.array([1, 1, 1]))

                point_invarient[patchNumber] = features[:-1]
                point_rotate[patchNumber] = features[-1]

                patchNumber += 1

    return point_invarient, point_rotate, patchNumber


@njit
def compute_orientation_population(point_rotate, patchNumber, ps):
    distance_population = np.empty(patchNumber)
    
    for i in range(patchNumber):
        ra = point_rotate[random.randint(0, patchNumber - 1)]
        rb = point_rotate[random.randint(0, patchNumber - 1)]
        max_tr = -10000
        for p in ps:
            max_tr = max(max_tr, np.trace(ra @ p @ rb.T))
        distance_population[i] = math.acos((max_tr - 1) / 2)
    return distance_population


def init_buckets(Q_TOTAL):
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS


def k_means_modeling(point_invarient, point_rotate):

    kmeans = KMeans_Tensor(n_clusters=C.Q_ANGLE, verbose=True, max_iter=30, n_init=1)
    kmeans.fit(point_invarient, point_rotate)

    return kmeans


def make_kmeans_model(file_list):
    G_WEIGHT = get_normalized_gaussian()

    MAX_POINTS = 15000000
    patchNumber = 0
    point_invarient = np.zeros((MAX_POINTS, 3), dtype='float32')
    point_rotate = np.zeros((MAX_POINTS, 3, 3), dtype='float32')

    for file_idx, file in enumerate(file_list):
        print('\r', end='')
        print('' * 60, end='')
        print('\r Making Point Space: '+ file.split('\\')[-1] + str(MAX_POINTS) + ' patches (' + str(100*patchNumber/MAX_POINTS) + '%)')

        im_HR, im_LR = get_array_data(file, training=True)
        im_GX, im_GY, im_GZ = np.gradient(im_LR)

        point_invarient, patch_rotate, patchNumber = make_point_space(im_LR, im_GX, im_GY, im_GZ, patchNumber, G_WEIGHT, point_invarient, point_rotate, MAX_POINTS)
        if patchNumber > MAX_POINTS / 2:
            break

    print(point_rotate.dtype)

    point_invarient = point_invarient[0:patchNumber, :]
    point_rotate = point_rotate[0:patchNumber, :]
    print(point_rotate.dtype)

    point_invarient = point_invarient * point_invarient
    std = np.std(point_invarient, axis=0)
    point_invarient = np.sqrt(point_invarient * (1/std)[None, :])

    # tensor orientation distance population
    
    # distance_population = np.empty(patchNumber)
    # ps = [np.diag([1, 1, 1]), np.diag([-1, -1, 1]), np.diag([-1, 1, -1]), np.diag([1, -1, -1])]
    # for i in range(patchNumber):
    #     ra = point_rotate[random.randint(0, patchNumber - 1)]
    #     rb = point_rotate[random.randint(0, patchNumber - 1)]
    #     max_tr = -10000
    #     for p in ps:
    #         max_tr = max(max_tr, np.trace(ra @ p @ rb.T))
    #     # tr_max = max(tr_max, max_tr)
    #     # tr_min = min(tr_min, max_tr)
    #     # print('\r', i, tr_max, tr_min, end='')
    #     if i % 100000 == 0:
    #         print('.', end='')
    #     distance_population[i] = math.acos((max_tr - 1) / 2)
    ps = [np.diag([1., 1., 1.]).astype('float32'), np.diag([-1., -1., 1.]).astype('float32'), np.diag([-1., 1., -1.]).astype('float32'), np.diag([1., -1., -1.]).astype('float32')]
    distance_population = compute_orientation_population(point_rotate, patchNumber, ps)
    std = np.append(std, np.std(distance_population))

    print(std)

    start = time.time()
    print('start clustering')
    kmeans = k_means_modeling(point_invarient, point_rotate)
    print(time.time() - start)

    with open('./arrays/space_{}x_{}.km'.format(C.R, C.Q_TOTAL), 'wb') as p:
        pickle.dump([kmeans, std], p)

    return kmeans, std


def load_kmeans_model():
    with open('./arrays/space_{}x_{}.km'.format(C.R, C.Q_TOTAL), "rb") as p:
        kmeans, std = pickle.load(p)
    return kmeans, std