import numpy as np
import os
import glob
import nibabel as nib
import math
import sys

from scipy.ndimage import zoom
from skimage.measure import compare_psnr, compare_ssim, compare_nrmse

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from crop_black import *
from filter_constant import *
from filter_func import *
from get_lr import *
from matrix_compute import *
from util import *


def quality_one(flag, file_num, result_dir, factor):
    A = '../NISR/test/T1w_acpc_dc_restore_brain_{}.nii.gz'.format(file_num)
    hr = nib.load(A).get_fdata()
    hr_crop = mod_crop(hr, factor)
    hr_clip = clip_image(hr_crop)
    H, W, D = hr_crop.shape

    hr2 = np.zeros(hr.shape)
    hr2[0:H, 0:W, 0:D] = hr_clip

    if flag:
        nn = nib.load('../NISR/test_lr/{}x_{}_nn.nii.gz'.format(factor, file_num)).get_fdata()
        tl = nib.load('../NISR/test_lr/{}x_{}_tl.nii.gz'.format(factor, file_num)).get_fdata()
        tc = nib.load('../NISR/test_lr/{}x_{}_tc.nii.gz'.format(factor, file_num)).get_fdata()


        print('\nnearest neighbor:\npsnr {}\nssim {}\nnrmse {}'.format(compare_psnr(hr2, nn, data_range=hr2.max()), compare_ssim(hr2, nn, data_range=hr2.max()), compare_nrmse(hr2, nn)))
        print('\ntrilinear:\npsnr {}\nssim {}\nnrmse {}'.format(compare_psnr(hr2, tl, data_range=hr2.max()), compare_ssim(hr2, tl, data_range=hr2.max()), compare_nrmse(hr2, tl)))
        print('\ntricubic:\npsnr {}\nssim {}\nnrmse {}'.format(compare_psnr(hr2, tc, data_range=hr2.max()), compare_ssim(hr2, tc, data_range=hr2.max()), compare_nrmse(hr2, tc)))

    rs = nib.load('{}/T1w_acpc_dc_restore_brain_{}_result.nii.gz'.format(result_dir, file_num)).get_fdata()  # [original.nonzero()]
    print('\nours:\npsnr {}\nssim {}\nnrmse {}'.format(compare_psnr(hr2, rs, data_range=hr2.max()), compare_ssim(hr2, rs, data_range=hr2.max()), compare_nrmse(hr2, rs)))

# True - nn/trilinear/tricubic 포함 여부
def quality_multiple(flag, result_dir, factor):
    files = [file.split('_')[-1].split('.')[0] for file in glob.glob('./test/*.nii.gz')]
    lst = [[] for i in range(12)]
    for idx, file in enumerate(files):
        print('\r', ' ' * 10, '\r', ' ' * 10, end='')
        print('\r{} / {}'.format(idx, len(files)), end='')
        A = '../NISR/test/T1w_acpc_dc_restore_brain_{}.nii.gz'.format(file)
        hr = nib.load(A).get_fdata()
        hr_crop = mod_crop(hr, factor)
        hr_clip = clip_image(hr_crop)
        H, W, D = hr_crop.shape

        hr2 = np.zeros(hr.shape)
        hr2[0:H, 0:W, 0:D] = hr_clip

        if flag:
            nn = nib.load('../NISR/test_lr/{}x_{}_nn.nii.gz'.format(factor, file)).get_fdata()
            tl = nib.load('../NISR/test_lr/{}x_{}_tl.nii.gz'.format(factor, file)).get_fdata()
            tc = nib.load('../NISR/test_lr/{}x_{}_tc.nii.gz'.format(factor, file)).get_fdata()

            lst[0].append(compare_psnr(hr2, nn, data_range=hr2.max()))
            lst[4].append(compare_ssim(hr2, nn, data_range=hr2.max()))
            lst[8].append(compare_nrmse(hr2, nn))

            lst[1].append(compare_psnr(hr2, tl, data_range=hr2.max()))
            lst[5].append(compare_ssim(hr2, tl, data_range=hr2.max()))
            lst[9].append(compare_nrmse(hr2, tl))

            lst[2].append(compare_psnr(hr2, tc, data_range=hr2.max()))
            lst[6].append(compare_ssim(hr2, tc, data_range=hr2.max()))
            lst[10].append(compare_nrmse(hr2, tc))

        print('.', end='', flush=True)
        rs = nib.load('{}/T1w_acpc_dc_restore_brain_{}_result.nii.gz'.format(result_dir, file)).get_fdata()

        lst[3].append(compare_psnr(hr2, rs, data_range=hr2.max()))
        lst[7].append(compare_ssim(hr2, rs, data_range=hr2.max()))
        lst[11].append(compare_nrmse(hr2, rs))

    if flag:
        print('\n--psnr--\nnn: {}+{}\ntrilinear: {}+{}\ntricubic: {}+{}\nours: {}+{}'.format( \
            np.mean(lst[0]), np.std(lst[0]), np.mean(lst[1]), np.std(lst[1]), \
            np.mean(lst[2]), np.std(lst[2]), np.mean(lst[3]), np.std(lst[3])))
        print('\n--ssim--\nnn: {}+{}\ntrilinear: {}+{}\ntricubic: {}+{}\nours: {}+{}'.format( \
            np.mean(lst[4]), np.std(lst[4]), np.mean(lst[5]), np.std(lst[5]), \
            np.mean(lst[6]), np.std(lst[6]), np.mean(lst[7]), np.std(lst[7])))
        print('\n--nrmse--\nnn: {}+{}\ntrilinear: {}+{}\ntricubic: {}+{}\nours: {}+{}'.format( \
            np.mean(lst[8]), np.std(lst[8]), np.mean(lst[9]), np.std(lst[9]), \
            np.mean(lst[10]), np.std(lst[10]), np.mean(lst[11]), np.std(lst[11])))
    else:
        print('\n--ours--\npsnr: {}+{}, ssim: {}+{}, nrmse: {}+{}'.format( \
            np.mean(lst[3]), np.std(lst[3]), np.mean(lst[7]), np.std(lst[7]), np.mean(lst[11]), np.std(lst[11])))


if __name__ == '__main__':
    quality_one(False, '894067', './result/111220', 4)
    # quality_one(False, '894067', './result/110315', 4)
    # quality_multiple(False, './result/110521', 4)
    
