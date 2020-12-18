import time
import random
import math

import cupy as cp
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans

import filter_constant as C

from feature_model import *
from filter_func import *
from kmeans_vector import KMeans_Vector
from matrix_compute import *
from preprocessing import *
from util import *


def train_qv(im_LR, im_HR, w, kmeans, std, Q, V, count):
    im_GX, im_GY, im_GZ = np.gradient(im_LR)

    point_list = sample_points(im_HR.shape)

    for sample_idx, point_list1 in enumerate(point_list):
        print('\r{} / {}'.format(sample_idx + 1, len(point_list)), end='', flush=True)
        patchS, xS = init_buckets()
        fS = []
        iS = []

        timer = time.time()

        for i1, j1, k1 in point_list1:
            if im_HR[i1, j1, k1] == 0:
                continue

            patchX, patchY, patchZ = get_gxyz(im_GX, im_GY, im_GZ, i1, j1, k1)
            features = get_features(patchX, patchY, patchZ, w, std)
            fS.append(features[:-2])
            iS.append(features[-2:])

        print('   get_feature {} s'.format(((time.time() - timer) * 1000 // 10) / 100), end='', flush=True)
        timer = time.time()
        fS = np.array(fS)
        jS_a = kmeans[0].predict(fS[:, :3])
        jS_t = kmeans[1].predict(fS[:, 3:])
        jS = jS_a + jS_t * C.Q_ANGLE
        cnt = 0

        print('   predict {} s'.format(((time.time() - timer) * 1000 // 10) / 100), end='', flush=True)
        timer = time.time()

        for i1, j1, k1 in point_list1:

            if im_HR[i1, j1, k1] == 0:
                continue

            patch = get_patch(im_LR, i1, j1, k1)
            patch = np.transpose(patch, iS[cnt][0])

            if iS[cnt][1][0] < 0:
                iS[cnt][1][1] *= -1
                iS[cnt][1][2] *= -1
            if iS[cnt][1][1] > 0 and iS[cnt][1][2] < 0:
                patch = np.flip(patch, axis=2)
            elif iS[cnt][1][1] < 0 and iS[cnt][1][2] > 0:
                patch = np.flip(patch, axis=1)
            elif iS[cnt][1][1] < 0 and iS[cnt][1][2] < 0:
                patch = np.flip(patch, axis=0)

            x1 = im_HR[i1, j1, k1]

            patchS[jS[cnt]].append(patch)
            xS[jS[cnt]].append(x1)
            count[jS[cnt]] += 1
            cnt += 1

        print('   append {} s'.format(((time.time() - timer) * 1000 // 10) / 100), end='', flush=True)
        timer = time.time()

        for j in range(C.Q_TOTAL):
            if len(xS[j]) != 0:
                A = np.array(patchS[j])
                A = A.reshape((A.shape[0], -1))
                A = np.concatenate((A, np.ones((A.shape[0], 1))), axis=1)
                b = np.array(xS[j]).reshape(-1, 1)
                Q[j] += np.dot(A.T, A)
                V[j] += np.dot(A.T, b).reshape(-1)

        print('   qv {} s'.format((time.time() - timer) * 100 // 10 / 10), end='', flush=True)

    return Q, V, count


if __name__ == '__main__':
    C.argument_parse()

    Q = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1, C.FILTER_VOL+1), dtype=np.float64)
    V = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1), dtype=np.float64)
    finished_files = []
    count = np.zeros(C.Q_TOTAL, dtype=int)

    file_list = make_dataset(C.TRAIN_DIR)
    C.TRAIN_FILE_MAX = min(C.TRAIN_FILE_MAX, len(file_list))

    G_WEIGHT = get_normalized_gaussian()

    # kmeans, std = make_kmeans_model(file_list)
    kmeans, std = load_kmeans_model()

    start = time.time()

    for file_idx, file in enumerate(file_list):
        file_name = file.split('\\')[-1].split('.')[0]
        filestart = time.time()

        if file_idx >= 100:
            break

        if file in finished_files:
            continue

        print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')')

        im_HR, im_LR = get_array_data(file, training=True)
        Q, V, count = train_qv(im_LR, im_HR, G_WEIGHT, kmeans, std, Q, V, count)
        
        print(' ' * 5, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

        finished_files.append(file)
        
    print(count)
    compute_h(Q, V)
