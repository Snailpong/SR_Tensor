import numpy as np
import cupy as cp
import math
import time

from numba import njit

from scipy.sparse.linalg import cg

import filter_constant as C

@njit
def get_patch(LR, xP, yP, zP):
    return LR[xP - C.PATCH_HALF: xP + (C.PATCH_HALF + 1),
              yP - C.PATCH_HALF: yP + (C.PATCH_HALF + 1),
              zP - C.PATCH_HALF: zP + (C.PATCH_HALF + 1)]


@njit
def get_gxyz(Lgx, Lgy, Lgz, xP, yP, zP):
    gx = Lgx[xP - C.GRADIENT_HALF: xP + (C.GRADIENT_HALF + 1),
             yP - C.GRADIENT_HALF: yP + (C.GRADIENT_HALF + 1),
             zP - C.GRADIENT_HALF: zP + (C.GRADIENT_HALF + 1)]
    gy = Lgy[xP - C.GRADIENT_HALF: xP + (C.GRADIENT_HALF + 1),
             yP - C.GRADIENT_HALF: yP + (C.GRADIENT_HALF + 1),
             zP - C.GRADIENT_HALF: zP + (C.GRADIENT_HALF + 1)]
    gz = Lgz[xP - C.GRADIENT_HALF: xP + (C.GRADIENT_HALF + 1),
             yP - C.GRADIENT_HALF: yP + (C.GRADIENT_HALF + 1),
             zP - C.GRADIENT_HALF: zP + (C.GRADIENT_HALF + 1)]
    return gx, gy, gz


def add_qv_jt(patchSa, xSa, Qa, Va, j, t):
    A = cp.array(patchSa)
    b = cp.array(xSa).reshape(-1, 1)

    Qa = cp.array(Qa)
    Va = cp.array(Va)

    Qa += cp.dot(A.T, A)
    Va += cp.dot(A.T, b)

    return Qa.get(), Va.get()


def compute_h(Q, V):
    h = np.zeros((Q.shape[0], Q.shape[1]))

    print("\rComputing H...   ")
    start = time.time()
    for j in range(C.Q_TOTAL):
        print('\r{} / {}'.format(j + 1, C.Q_TOTAL), end='')
        h[j] = cg(Q[j], V[j], tol=1e-5)[0]

    h = np.array(h, dtype=np.float32)
    np.save('./arrays/h_{}x_{}'.format(C.R, C.Q_TOTAL), h)