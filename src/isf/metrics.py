import pandas as pd
import numpy as np
import scipy as sp
from math import sqrt
from sklearn.neighbors import KDTree


'''
Calculate the L2-norm distance
'''
def calculate_l2_dist(x1, x2):
    return np.linalg.norm(x1 - x2, 2)


'''
Calculate Raw Sparsity : Number of non-similar elements(feature differences) between two arrays (SF and Query)
'''
def calculate_sparsity(x1, x2):
    non_zero_threshold_sparsity = 1e-5 #0.00001
    diff = x1 - x2
    return np.sum(np.abs(diff[i]) > non_zero_threshold_sparsity for i in range(diff.shape[0]))


'''
Calculate OOD Distribution measure using E Kenny's SF paper
- Distance from the SF to the nearest training datapoint
'''
def calculate_ood(nbrs, y_train, pred, sf):
    # check if nearest neighbor is the same class as the SF
    nn = True
    # get distances and indices for neighbors of the computed SF
    distances, indices = nbrs.kneighbors(np.reshape(sf, (1, -1)))

    if nn:
        idxs = indices[0]
        dists = distances[0]
        for i in range(len(idxs)):
            if (y_train[idxs[i]] == pred):
                return dists[i], idxs[i]
    else:
        return distances[0][1], indices[0][1]


'''
Trust Score based on https://github.com/google/TrustScore/blob/master/trustscore/trustscore.py
'''


class TrustScore:
    """
    Trust Score: a measure of classifier uncertainty based on nearest neighbors.
  """

    def __init__(self, k=10, alpha=0.1, filtering="density", min_dist=1e-12):
        """
        k and alpha are the tuning parameters for the filtering,
        filtering: method of filtering. option are "none", "density",
        "uncertainty"
        min_dist: some small number to mitigate possible division by 0.
    """
        self.k = k
        self.filtering = filtering
        self.alpha = alpha
        self.min_dist = min_dist

    def filter_by_density(self, X: np.array):
        """Filter out points with low kNN density.

    Args:
    X: an array of sample points.

    Returns:
    A subset of the array without points in the bottom alpha-fraction of
    original points of kNN density.
    """
        kdtree = KDTree(X)
        knn_radii = kdtree.query(X, k=self.k)[0][:, -1]
        eps = np.percentile(knn_radii, (1 - self.alpha) * 100)
        return X[np.where(knn_radii <= eps)[0], :]

    def fit(self, X: np.array, y: np.array):
        """Initialize trust score precomputations with training data.

    WARNING: assumes that the labels are 0-indexed (i.e.
    0, 1,..., n_labels-1).

    Args:
    X: an array of sample points.
    y: corresponding labels.
    """

        self.n_labels = np.max(y) + 1
        self.kdtrees = [None] * self.n_labels
        for label in range(self.n_labels):
            if self.filtering == "none":
                X_to_use = X[np.where(y == label)[0]]
                self.kdtrees[label] = KDTree(X_to_use)
            elif self.filtering == "density":
                X_to_use = self.filter_by_density(X[np.where(y == label)[0]])
                self.kdtrees[label] = KDTree(X_to_use)

    def get_score(self, X: np.array, y_pred: np.array):
        """Compute the trust scores.

    Given a set of points, determines the distance to each class.

    Args:
    X: an array of sample points.
    y_pred: The predicted labels for these points.

    Returns:
    The trust score, which is ratio of distance to closest class that was not
    the predicted class to the distance to the predicted class.
    """
        d = np.tile(None, (X.shape[0], self.n_labels))
        for label_idx in range(self.n_labels):
            d[:, label_idx] = self.kdtrees[label_idx].query(X, k=2)[0][:, -1]

        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), y_pred]
        d_to_closest_not_pred = np.where(
            sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1]
        )
        return d_to_closest_not_pred / (d_to_pred + self.min_dist)

