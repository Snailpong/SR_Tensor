import numpy as np
import scipy.sparse as sp

import sklearn
from sklearn.utils import check_random_state
from sklearn.utils.validation import _check_sample_weight
from sklearn.cluster._kmeans import _init_centroids
from sklearn.utils.extmath import row_norms

from numba import njit, prange


class KMeans_Vector(sklearn.cluster.KMeans):
    def __init__(self, n_clusters=8, *, n_init=10, max_iter=300, tol=1e-4, verbose=0):
        super().__init__(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter, tol=tol, verbose=verbose)


    def fit(self, X, y=None, sample_weight=None):
        X = super()._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)

        super()._check_params(X)
        random_state = check_random_state(self.random_state)

        init = self.init
        x_squared_norms = row_norms(X, squared=True)
        best_labels, best_inertia, best_centers = None, None, None
        seeds = random_state.randint(np.iinfo(np.int32).max, size=self._n_init)

        for seed in seeds:
            labels, inertia, centers, n_iter_ = kmeans_single_elkan_vector(
                X, sample_weight, self.n_clusters, max_iter=self.max_iter,
                init=init, verbose=self.verbose, tol=self._tol,
                x_squared_norms=x_squared_norms, random_state=seed,
                n_threads=self._n_threads)

            if best_inertia is None or inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
                best_n_iter = n_iter_

        distinct_clusters = len(set(best_labels))

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    
    def predict(self, X):
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=np.int32)
        labels = get_predict(self.cluster_centers_, labels, n_samples, X)

        return labels


@njit
def get_predict(centers, labels, n_samples, X):
    for sample in range(n_samples):
        max_similarity = -1
        for l in range(centers.shape[0]):
            similarity = np.dot(X[sample], centers[l])
            if max_similarity < similarity:
                max_similarity = similarity
                labels[sample] = l

    return labels


@njit
def get_new_centers(centers, centers_new, labels, n_samples, X):
    for sample in range(n_samples):
        max_similarity = -1
        for l in range(centers.shape[0]):
            similarity = np.dot(X[sample], centers[l])
            if max_similarity < similarity:
                max_similarity = similarity
                labels[sample] = l
        centers_new[labels[sample]] += X[sample]

    return centers_new, labels


def kmeans_single_elkan_vector(X, sample_weight, n_clusters, max_iter=300,
                         init='k-means++', verbose=False, x_squared_norms=None,
                         random_state=None, tol=1e-4, n_threads=1):

    random_state = check_random_state(random_state)
    sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

    centers = _init_centroids(X, n_clusters, init, random_state=random_state,
                              x_squared_norms=x_squared_norms)

    centers_norm = np.linalg.norm(centers, axis=1)
    centers = centers / centers_norm[:, None]

    if verbose:
        print('Initialization complete')

    n_samples = X.shape[0]

    labels = np.full(n_samples, -1, dtype=np.int32)
    labels_old = labels.copy()

    for i in range(max_iter):
        centers_new = np.zeros_like(centers)

        centers_new, labels = get_new_centers(centers, centers_new, labels, n_samples, X)

        centers_norm_new = np.linalg.norm(centers_new, axis=1)
        centers_new = centers_new / centers_norm_new[:, None]

        if np.array_equal(labels, labels_old):
            if verbose:
                print(f"Converged at iteration {i}: strict convergence.")
            strict_convergence = True
            break

        center_shift = centers_new - centers
        center_shift_tot = (center_shift**2).sum()

        if center_shift_tot <= tol:
            if verbose:
                print(f"Converged at iteration {i}: center shift "
                        f"{center_shift_tot} within tolerance {tol}.")
            break

        if verbose:
            print("Iteration {0}, center_shift {1}" .format(i, center_shift_tot))

        labels_old = labels.copy()
        centers = centers_new.copy()
        
    return labels, center_shift_tot, centers, i + 1

