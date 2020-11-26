import os
import glob
import time

import numpy as np
import cupy as cp
from numba import jit, njit, prange

import nibabel as nib

import filter_constant as C

from crop_black import *
from filter_constant import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from train import load_kmeans_model, get_features

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def make_image(im_LR, im_GX, im_GY, im_GZ, w, kmeans, h):
    H = im_LR.shape[0]
    result_image = im_LR.copy()

    timer = time.time()
    for i1 in range(C.PATCH_HALF, H - C.PATCH_HALF):
        print('\r{} / {}    {} s'.format(i1, H - C.PATCH_HALF, ((time.time() - timer) * 100 // 10) / 10), end='')
        timer = time.time()
        fS, iS = get_feature_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w)

        if len(fS) == 0:
            continue

        jS = kmeans.predict(fS)
        result_image = make_hr_yz(i1, result_image, im_LR, jS, h, iS)

    result_image = np.clip(result_image, 0, 1)

    return result_image


def get_feature_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w):
    H, W, D = im_LR.shape
    fS = []
    iS = []
    
    for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
        for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):
            if im_LR[i1, j1, k1] == 0:
                continue

            idxg = (slice(i1 - C.GRADIENT_HALF, i1 + C.GRADIENT_HALF + 1),
                    slice(j1 - C.GRADIENT_HALF, j1 + C.GRADIENT_HALF + 1),
                    slice(k1 - C.GRADIENT_HALF, k1 + C.GRADIENT_HALF + 1))

            patchX = im_GX[idxg]
            patchY = im_GY[idxg]
            patchZ = im_GZ[idxg]

            features = get_features(patchX, patchY, patchZ, w)
            fS.append(features[:-2])
            iS.append(features[-2:])

    return fS, iS


# @jit
def make_hr_yz(i1, result_image, im_LR, jS, h, iS):
    H, W, D = im_LR.shape
    cnt = 0
    
    for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
        for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):
            idxp = (slice(i1 - C.PATCH_HALF, i1 + C.PATCH_HALF + 1),
                    slice(j1 - C.PATCH_HALF, j1 + C.PATCH_HALF + 1),
                    slice(k1 - C.PATCH_HALF, k1 + C.PATCH_HALF + 1))
            patch = im_LR[idxp]

            if im_LR[i1, j1, k1] == 0:
                continue

            # patch1 = patch.ravel()
            # print(iS[cnt][0])
            patch = np.transpose(patch, iS[cnt][0])

            if iS[cnt][1][0] > 0 and iS[cnt][1][1] < 0:
                patch = np.flip(patch, axis=2)
            elif iS[cnt][1][0] < 0 and iS[cnt][1][1] > 0:
                patch = np.flip(patch, axis=1)
            elif iS[cnt][1][0] < 0 and iS[cnt][1][1] < 0:
                patch = np.flip(patch, axis=0)

            patch1 = np.append(patch, 1).astype(np.float32)
            result_image[i1, j1, k1] = np.dot(patch1, h[jS[cnt]])
            cnt += 1

    return result_image


if __name__ == '__main__':
    C.argument_parse()
    C.Q_TOTAL = 256

    current_hour = time.strftime('%m%d%H', time.localtime(time.time()))
    result_dir = './result/{}_{}x_{}/'.format(current_hour, C.R, C.Q_TOTAL)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    file_list = make_dataset('../NISR/test')

    # Preprocessing normalized Gaussian matrix W for hashkey calculation
    G_WEIGHT = get_normalized_gaussian()

    h = np.load('./arrays/h_{}x_{}.npy'.format(C.R, C.Q_TOTAL))
    kmeans = load_kmeans_model()

    filestart = time.time()

    for file_idx, file in enumerate(file_list):
        file_name = file.split('\\')[-1].split('.')[0]
        print('\r', end='')
        print('' * 60, end='')
        print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')' + str(time.time() - filestart))
        filestart = time.time()

        raw_image = np.array(nib.load(file).get_fdata(), dtype=np.float32)
        clipped_image = clip_image(raw_image)
        im = mod_crop(clipped_image, C.R)
        slice_area = crop_slice(im, C.PATCH_SIZE // 2, C.R)

        im_HR = im[slice_area] / im.max()
        im_blank_LR = get_lr(im) / im.max()
        im_LR = im_blank_LR[slice_area]
        im_GX, im_GY, im_GZ = np.gradient(im_LR)

        im_result = make_image(im_LR, im_GX, im_GY, im_GZ, G_WEIGHT, kmeans, h)

        output_img = np.zeros(raw_image.shape)
        output_img[slice_area] = im_result
        output_img = output_img * im.max()
        ni_img = nib.Nifti1Image(output_img, np.eye(4))
        nib.save(ni_img, '{}/{}_result.nii.gz'.format(result_dir, file_name))

        print()
        print(peak_signal_noise_ratio(im_HR, im_LR), peak_signal_noise_ratio(im_HR, im_result))
        print(structural_similarity(im_HR, im_LR), structural_similarity(im_HR, im_result))

        if file_idx == 0:
            break

    print("Test is off")