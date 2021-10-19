import numpy as np
import math
import nibabel as nib
import random
import scipy as sp

import filter_constant as C


def get_array_data(file, training):
    raw_image = nib.load(file)
    raw_array = np.array(raw_image.get_fdata(), dtype=np.float32)
    raw_header = raw_image.header.copy()

    clipped_image = clip_image(raw_array)
    im = mod_crop(clipped_image, C.R)
    slice_area = crop_slice(im, C.PATCH_HALF, C.R)

    im_blank_LR = get_lr(im) / im.max()
    im_LR = im_blank_LR[slice_area]
    im_HR = im[slice_area] / im.max()

    if training:
        return im_HR, im_LR
    else:
        return im_HR, im_LR, raw_image.shape, im.max(), slice_area, raw_header


def mod_crop(im, modulo):
    H, W, D = im.shape
    size0 = H - H % modulo
    size1 = W - W % modulo
    size2 = D - D % modulo

    out = im[0:size0, 0:size1, 0:size2]

    return out


def clip_image(im):
    clip_value = np.sort(im.ravel())[int(np.prod(im.shape) * 0.999)]
    im = np.clip(im, 0, clip_value)
    return im
    

def get_lr(hr):
    if C.LR_TYPE == 'interpolation':
        lr = get_lr_interpolation(hr)   # Using Image domain
    else:
        lr = get_lr_kspace(hr)          # Using Frequency domain
    return lr
    

def get_lr_interpolation(im):
    downscaled_lr = sp.ndimage.zoom(im, 1.0 / C.R, order=2, prefilter=False)
    lr = np.clip(sp.ndimage.zoom(downscaled_lr, C.R, order=2, prefilter=False), 0, im.max())
    lr[np.where(im == 0)] = 0
    return lr


def get_lr_kspace(hr):
    imgfft = np.fft.fftn(hr)
    imgfft_zero = np.zeros((imgfft.shape[0], imgfft.shape[1], imgfft.shape[2]))

    x_area = y_area = z_area = 50

    x_center = imgfft.shape[0] // 2
    y_center = imgfft.shape[1] // 2
    z_center = imgfft.shape[2] // 2

    imgfft_shift = np.fft.fftshift(imgfft)
    imgfft_shift2 = imgfft_shift.copy()

    imgfft_shift[x_center-x_area : x_center+x_area, y_center-y_area : y_center+y_area, z_center-z_area : z_center+z_area] = 0
    imgfft_shift2 = imgfft_shift2 - imgfft_shift

    imgifft3 = np.fft.ifftn(imgfft_shift2)
    lr = abs(imgifft3)
    return lr


def chunk(lst, size):
    return list(map(lambda x: lst[x * size:x * size + size], list(range(0, math.ceil(len(lst) / size)))))


def sample_points(array_shape):
    h, w, d = array_shape
    xyz_range = [[x, y, z] for x in range(C.PATCH_HALF, h - C.PATCH_HALF)
                        for y in range(C.PATCH_HALF, w - C.PATCH_HALF)
                        for z in range(C.PATCH_HALF, d - C.PATCH_HALF)]
    sample_range = random.sample(xyz_range, len(xyz_range) // C.SAMPLE_RATE)
    point_list = chunk(sample_range, len(sample_range) // C.TRAIN_DIV + 1)

    return point_list


def crop_slice(array, padding, factor):
    for i in range(padding, array.shape[0] - padding):
        if not np.all(array[i, :, :] == 0):
            x_use1 = i - padding
            x_use1 = x_use1 - (x_use1 % factor)
            break
    for i in reversed(range(padding, array.shape[0] - padding)):
        if not np.all(array[i, :, :] == 0):
            x_use2 = i + padding
            break
    for i in range(padding, array.shape[1] - padding):
        if not np.all(array[:, i, :] == 0):
            y_use1 = i - padding
            y_use1 = y_use1 - (y_use1 % factor)
            break
    for i in reversed(range(padding, array.shape[1] - padding)):
        if not np.all(array[:, i, :] == 0):
            y_use2 = i + padding
            break
    for i in range(padding, array.shape[2] - padding):
        if not np.all(array[:, :, i] == 0):
            z_use1 = i - padding
            z_use1 = z_use1 - (z_use1 % factor)
            break
    for i in reversed(range(padding, array.shape[2] - padding)):
        if not np.all(array[:, :, i] == 0):
            z_use2 = i + padding
            break

    area = (slice(x_use1, x_use2), slice(y_use1, y_use2), slice(z_use1, z_use2))
    return area
