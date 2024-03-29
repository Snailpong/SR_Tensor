import numpy as np
import os
import pickle
import nibabel as nib

from filter_func import *
from matrix_compute import *
from util import *

import filter_constant as C


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for fname in sorted(os.listdir(dir)):
        if fname.endswith('.nii.gz'):
            path = os.path.join(dir, fname)
            images.append(path)
    return images


def save_qv(Q, V, finished_files, count):
    print('\rSaving QVF...', end='', flush=True)
    np.savez('./arrays/QVF_{}'.format(C.R), Q=Q, V=V, finished_files=np.array(finished_files), count=count)


def init_buckets():
    patchS = [[] for j in range(C.Q_TOTAL)]
    xS = [[] for j in range(C.Q_TOTAL)]
    return patchS, xS


def load_files():
    if os.path.isfile('./arrays/QVF_{}.npz'.format(C.R)):
        print('Loading QVF...', end=' ', flush=True)
        QVF = np.load('./arrays/QVF_{}.npz'.format(C.R))
        Q = QVF['Q']
        V = QVF['V']
        finished_files = QVF['finished_files'].tolist()
        count = QVF['count']
        QVF.close()
        print('Done', flush=True)
    else:
        Q = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1, C.FILTER_VOL+1), dtype=np.float64)
        V = np.zeros((C.Q_TOTAL, C.FILTER_VOL+1), dtype=np.float64)
        finished_files = []
        count = np.zeros(C.Q_TOTAL, dtype=int)

    return Q, V, finished_files, count
