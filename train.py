import glob
import time
import random
import math

import cupy as cp
import numpy as np

import nibabel as nib
from sklearn.cluster import KMeans

import filter_constant as C

from crop_black import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from util import *
from kmeans_vector import KMeans_Vector


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

    angle_p = math.atan2(v1[1], v1[0])
    angle_t = math.acos(v1[2] / (sqrt((v1[0]) ** 2 + v1[1] ** 2 + v1[2] ** 2) + 1e-16))

    trace, fa, mode = get_lamda_u(l1, l2, l3)

    # return angle_p, angle_t, math.log(trace), fa, mode
    # return angle_p, angle_t, math.log(trace)/4, fa, mode/2, index1, sign
    return v1[0], v1[1], v1[2], math.log(trace), fa, mode, index1, sign


@njit
def make_point_space(im_LR, im_GX, im_GY, im_GZ, patchNumber, w, point_space, MAX_POINTS):
    H, W, D = im_GX.shape

    for i1 in range(C.PATCH_HALF, H - C.PATCH_HALF):
        # print(i1)
        for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
            for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):

                # if random.random() > 0.2 or np.any(im_LR[i1, j1, k1] == 0):
                #     continue

                if im_LR[i1, j1, k1] == 0:
                    continue

                patchX, patchY, patchZ = get_gxyz(im_GX, im_GY, im_GZ, i1, j1, k1)

                point_space[patchNumber] = np.array(get_features(patchX, patchY, patchZ, w)[:-2])
                patchNumber += 1

                # print(point_space[patchNumber-1, 0:2])

    return point_space, patchNumber


def k_means_modeling(quantization):

    with open('./arrays/qua', 'rb') as p:
        quantization = pickle.load(p)

    kmeans_angle = KMeans_Vector(n_clusters=C.Q_ANGLE, verbose=True, max_iter=30, n_init=1)
    kmeans_angle.fit(quantization[:, :3])

    kmeans_tensor = KMeans(n_clusters=C.Q_TENSOR, verbose=True, max_iter=30, n_init=1)
    kmeans_tensor.fit(quantization[:, 3:])

    return kmeans_angle, kmeans_tensor


def train_qv2(im_LR, im_HR, w, kmeans, Q, V, count):
    H, W, D = im_HR.shape
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images

    xyz_range = [[x, y, z] for x in range(C.PATCH_HALF, H - C.PATCH_HALF)
                    for y in range(C.PATCH_HALF, W - C.PATCH_HALF)
                    for z in range(C.PATCH_HALF, D - C.PATCH_HALF)]
    sample_range = random.sample(xyz_range, len(xyz_range) // C.SAMPLE_RATE)
    point_list = chunk(sample_range, len(sample_range) // C.TRAIN_DIV + 1)

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
            features = get_features(patchX, patchY, patchZ, w)
            fS.append(features[:-2])
            iS.append(features[-2:])

        print('   get_feature {} s'.format(((time.time() - timer) * 1000 // 10) / 100), end='', flush=True)
        timer = time.time()
        fS = np.array(fS)
        jS_a = kmeans[0].predict(fS[:, :2])
        jS_t = kmeans[1].predict(fS[:, 2:])
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

            # patch1 = append_func(patch)
            patch1 = np.append(patch, 1)
            x1 = im_HR[i1, j1, k1]

            patchS[jS[cnt]].append(patch1)
            xS[jS[cnt]].append(x1)
            count[jS[cnt]] += 1
            cnt += 1

        print('   append {} s'.format(((time.time() - timer) * 1000 // 10) / 100), end='', flush=True)
        timer = time.time()

        for j in range(C.Q_TOTAL):
            if len(xS[j]) != 0:
                A = np.array(patchS[j])
                b = np.array(xS[j]).reshape(-1, 1)
                Q[j] += np.dot(A.T, A)
                V[j] += np.dot(A.T, b).reshape(-1)

        print('   qv {} s'.format((time.time() - timer) * 100 // 10 / 10), end='', flush=True)

    return Q, V, count


def make_kmeans_model():
    G_WEIGHT = get_normalized_gaussian()

    MAX_POINTS = 15000000
    patchNumber = 0
    point_space = np.zeros((MAX_POINTS, 6))

    # for file_idx, file in enumerate(file_list):
    #     print('\r', end='')
    #     print('' * 60, end='')
    #     print('\r Making Point Space: '+ file.split('\\')[-1] + str(MAX_POINTS) + ' patches (' + str(100*patchNumber/MAX_POINTS) + '%)')

    #     im_HR, im_LR = get_array_data(file, training=True)
    #     im_GX, im_GY, im_GZ = np.gradient(im_LR)

    #     point_space, patchNumber = make_point_space(im_LR, im_GX, im_GY, im_GZ, patchNumber, G_WEIGHT, point_space, MAX_POINTS)
    #     if patchNumber > MAX_POINTS / 2:
    #         break

    quantization = point_space[0:patchNumber, :]

    # start = time.time()
    print('start clustering')
    kmeans = k_means_modeling(quantization)
    print(time.time() - start)

    with open('./arrays/space_{}x_{}.km'.format(C.R, C.Q_TOTAL), 'wb') as p:
        pickle.dump(kmeans, p)

    return kmeans


def load_kmeans_model():
    with open('./arrays/space_{}x_{}.km'.format(C.R, C.Q_TOTAL), "rb") as p:
        kmeans = pickle.load(p)
    return kmeans


if __name__ == '__main__':
    C.argument_parse()

    Q = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1, C.FILTER_VOL+1), dtype=np.float64)
    V = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1), dtype=np.float64)
    finished_files = []
    count = np.zeros(C.Q_TOTAL, dtype=int)

    file_list = make_dataset(C.TRAIN_DIR)
    C.TRAIN_FILE_MAX = min(C.TRAIN_FILE_MAX, len(file_list))

    # Preprocessing normalized Gaussian matrix W for hashkey calculation
    G_WEIGHT = get_normalized_gaussian()

    kmeans = make_kmeans_model()
    # kmeans = load_kmeans_model()

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
        Q, V, count = train_qv2(im_LR, im_HR, G_WEIGHT, kmeans, Q, V, count)
        
        print(' ' * 5, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

        finished_files.append(file)
        
    print(count)
    compute_h(Q, V)
