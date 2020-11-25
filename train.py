import glob
import time
import random

import cupy as cp
import numpy as np

import nibabel as nib

import filter_constant as C

from crop_black import *
from filter_func import *
from get_lr import *
from hashtable import *
from matrix_compute import *
from util import *

C.argument_parse()

# Q, V, finished_files, count = load_files()
Q = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1, C.FILTER_VOL+1), dtype=np.float64)
V = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1), dtype=np.float64)
finished_files = []
count = np.zeros(C.Q_TOTAL, dtype=int)

b_trace = np.zeros((C.Q_TRACE - 1)) 
b_fa = np.zeros((C.Q_FA - 1)) 
b_mode = np.zeros((C.Q_MODE - 1)) 

file_list = make_dataset('../NISR/train')
C.TRAIN_FILE_MAX = min(C.TRAIN_FILE_MAX, len(file_list))

# Preprocessing normalized Gaussian matrix W for hashkey calculation
G_WEIGHT = get_normalized_gaussian()

# instance = 5000000
# patchNumber = 0
# quantization = np.zeros((instance, 3))
# for file_idx, image in enumerate(file_list):
#     print('\r', end='')
#     print('' * 60, end='')
#     print('\r Quantization: Processing '+ image.split('\\')[-1] + str(instance) + ' patches (' + str(100*patchNumber/instance) + '%)')

#     raw_image = np.array(nib.load(image).get_fdata(), dtype=np.float32)
#     clipped_image = clip_image(raw_image)
#     im = mod_crop(clipped_image, C.R)

#     slice_area = crop_slice(im, C.PATCH_SIZE // 2, C.R)
#     im_LR = get_lr(im)

#     im_blank_LR = get_lr(im) / im.max()
#     im_LR = im_blank_LR[slice_area]
#     im_GX, im_GY, im_GZ = np.gradient(im_LR)  # Calculate the gradient images

#     quantization, patchNumber = quantization_border(im_LR, im_GX, im_GY, im_GZ, patchNumber, G_WEIGHT, quantization, instance)  # get the strength and coherence of each patch
#     if patchNumber > instance / 2:
#         break

# # uniform quantization of patches, get the optimized strength and coherence boundaries
# quantization = quantization[0:patchNumber, :]
# quantization = np.sort(quantization, axis=0)

# for i in range(C.Q_TRACE - 1):
#     b_trace[i] = quantization[floor((i+1) * patchNumber / C.Q_TRACE), 0]
#     b_fa[i] = quantization[floor((i+1) * patchNumber / C.Q_FA), 1]
#     b_mode[i] = quantization[floor((i+1) * patchNumber / C.Q_MODE), 2]

b_trace = np.array([0.03283301, 0.05263998])
b_fa = np.array([0.36205414, 0.47029916])
b_mode = np.array([0.84320635, 0.96727021])

# b_trace = np.array([0.05, 0.15])
# b_fa = np.array([0.35, 0.45])
# b_mode = np.array([0, 0.75])

print(b_trace, b_fa, b_mode)

start = time.time()

for file_idx, file in enumerate(file_list):
    file_name = file.split('\\')[-1].split('.')[0]
    filestart = time.time()

    if file_idx >= 100:
        break

    if file in finished_files:
        continue

    print('\rProcessing ' + str(file_idx + 1) + '/' + str(len(file_list)) + ' image (' + file_name + ')')

    raw_image = np.array(nib.load(file).get_fdata(), dtype=np.float32)
    clipped_image = clip_image(raw_image)
    im = mod_crop(clipped_image, C.R)
    slice_area = crop_slice(im, C.PATCH_HALF, C.R)

    im_blank_LR = get_lr(im) / im.max()  # Prepare the cheap-upscaling images
    im_LR = im_blank_LR[slice_area]
    im_HR = im[slice_area] / im.max()

    Q, V, count = train_qv2(im_LR, im_HR, G_WEIGHT, b_trace, b_fa, b_mode, Q, V, count)  # get Q, V of each patch
    
    print(' ' * 30, 'last', '%.1f' % ((time.time() - filestart) / 60), 'min', end='', flush=True)

    finished_files.append(file)

compute_h(Q, V)

with open("./arrays/Qfactor_" + str(C.R), "wb") as p:
    pickle.dump([b_trace, b_fa, b_mode], p)