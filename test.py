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

from skimage.measure import compare_psnr, compare_ssim

C.argument_parse()


def make_image(im_LR, im_GX, im_GY, im_GZ, w, b_trace, b_fa, b_mode, h):
    H = im_LR.shape[0]
    result_image = im_LR.copy()
    # im_LR = np.array(im_LR, dtype=np.float64)

    timer = time.time()
    for i1 in range(C.PATCH_HALF, H - C.PATCH_HALF):
        print('\r{} / {}    {} s'.format(i1, H - C.PATCH_HALF, ((time.time() - timer) * 100 // 10) / 10), end='')
        timer = time.time()
        result_image = make_image_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w, b_trace, b_fa, b_mode, h)

    result_image = np.clip(result_image, 0, 1)

    return result_image


@njit
def make_image_yz(i1, result_image, im_LR, im_GX, im_GY, im_GZ, w, b_trace, b_fa, b_mode, h):
    H, W, D = im_LR.shape
    
    for j1 in range(C.PATCH_HALF, W - C.PATCH_HALF):
        for k1 in range(C.PATCH_HALF, D - C.PATCH_HALF):
            idxp = (slice(i1 - C.PATCH_HALF, i1 + C.PATCH_HALF + 1),
                    slice(j1 - C.PATCH_HALF, j1 + C.PATCH_HALF + 1),
                    slice(k1 - C.PATCH_HALF, k1 + C.PATCH_HALF + 1))
            patch = im_LR[idxp]

            if im_LR[i1, j1, k1] == 0:
                continue

            # np.where(patch == 0, patch[C.PATCH_HALF, C.PATCH_HALF, C.PATCH_HALF], patch)

            # if np.any(patch == 0):
            #     patch[np.where(patch == 0)] = patch[PATCH_HALF, PATCH_HALF, PATCH_HALF]

            idxg = (slice(i1 - C.GRADIENT_HALF, i1 + C.GRADIENT_HALF + 1),
                    slice(j1 - C.GRADIENT_HALF, j1 + C.GRADIENT_HALF + 1),
                    slice(k1 - C.GRADIENT_HALF, k1 + C.GRADIENT_HALF + 1))

            # patch_std = patch.std()
            patchX = im_GX[idxg]
            patchY = im_GY[idxg]
            patchZ = im_GZ[idxg]

            angle_p, angle_t, trace, fa, mode = get_hash(patchX, patchY, patchZ, w, b_trace, b_fa, b_mode)
            j = int(angle_p * C.Q_TRACE * C.Q_FA * C.Q_MODE * C.Q_ANGLE_T + angle_t * C.Q_TRACE * C.Q_FA * C.Q_MODE + trace * C.Q_FA * C.Q_MODE + fa * C.Q_MODE + mode)

            # patch1 = patch.ravel()
            patch1 = np.append(patch, 1).astype(np.float32)
            # patch1.append(1)
            result_image[i1, j1, k1] = np.dot(patch1, h[j])

    return result_image


current_hour = time.strftime('%m%d%H', time.localtime(time.time()))
result_dir = './result/{}/'.format(current_hour)
if not os.path.isdir(result_dir):
    os.mkdir(result_dir)

file_list = make_dataset('../NISR/test')

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

h = np.load('./arrays/h_{}.npy'.format(C.R))

# cnt = 0
# for i in range(h.shape[0]):
#     if h[i].max() >= 10 or h[i].min() <= -10:
#         h[i] = np.zeros((h.shape[1]))
#         h[i, C.PATCH_HALF * (C.PATCH_SIZE ** 2 + C.PATCH_SIZE) + C.PATCH_HALF] = 1
#         cnt += 1
# print(cnt)

with open("./arrays/Qfactor_" + str(C.R), "rb") as p:
    b_trace, b_fa, b_mode = pickle.load(p)

print(b_trace, b_fa, b_mode)
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
    im_blank_LR = get_lr(im) / im.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images

    im_result = make_image(im_LR, im_GX, im_GY, im_GZ, G_WEIGHT, b_trace, b_fa, b_mode, h)

    output_img = np.zeros(raw_image.shape)
    output_img[slice_area] = im_result
    output_img = output_img * im.max()
    ni_img = nib.Nifti1Image(output_img, np.eye(4))
    nib.save(ni_img, '{}/{}_result.nii.gz'.format(result_dir, file_name))

    # output_img2 = np.zeros(raw_image.shape)
    # output_img2[slice_area] = im_blending
    # output_img2 = output_img2 * clipped_image.max()
    # ni_img2 = nib.Nifti1Image(output_img2, np.eye(4))
    # nib.save(ni_img2, '{}/{}_result_blend.nii.gz'.format(result_dir, file_name))

    print()
    print(compare_psnr(im_HR, im_LR), compare_psnr(im_HR, im_result))
    print(compare_ssim(im_HR, im_LR), compare_ssim(im_HR, im_result))

    #if file_idx == 0:
        #break



print("Test is off")