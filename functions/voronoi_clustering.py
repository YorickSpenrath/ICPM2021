from abc import ABC, abstractmethod

import numpy as np
from numba import njit

data_type_feature_values = 'feature_values'
datatype_distance_matrix = 'distance_matrix'


# Methods --------------------------------------------------------------------------------------------------------------
def numba_voronoi_matrix(distance_matrix, im_initial, multiplicity=None, **kwargs):
    return _NumbaVoronoiMatrix(**kwargs) \
        .cluster(data=distance_matrix, im_initial=im_initial, multiplicity=multiplicity)


# Abstract Base --------------------------------------------------------------------------------------------------------
class _NumbaVoronoi(ABC):
    def __init__(self, **kwargs):
        self.eps = kwargs.pop('eps', 0.001)
        self.max_iter = kwargs.pop('max_iter', 1000000)
        if kwargs:
            raise ValueError(f'Got unknown input(s): {kwargs}')

    def cluster(self, data, im_initial, multiplicity=None):
        assert isinstance(data, np.ndarray) and len(data.shape) == 2, 'data should be 2d numpy array'
        n_samples = data.shape[0]
        if multiplicity is None:
            multiplicity = np.ones(n_samples, )
        return self._cluster(data, im_initial, multiplicity)

    @abstractmethod
    def _cluster(self, data, im_initial, multiplicity):
        pass


# Matrix based ---------------------------------------------------------------------------------------------------------
class _NumbaVoronoiMatrix(_NumbaVoronoi):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _cluster(self, data, im_initial, multiplicity):
        assert data.shape[1] == data.shape[0]
        im_old = im_initial.copy()
        n_medoids = len(im_initial)
        n_samples = data.shape[0]

        iteration_number = 0

        while True:
            iteration_number += 1

            # Find clusters
            labels = nv_matrix_update_clusters(data, im_old, n_samples)

            # Update medoids
            im_new = nv_matrix_update_medoids(labels, data, n_medoids, multiplicity)

            # Check if end
            if nv_matrix_stop(iteration_number, self.max_iter, data, im_old, im_new, n_medoids, self.eps):
                break
            else:
                im_old = im_new

        return im_new, labels


@njit
def nv_matrix_update_clusters(distance_matrix, indices_of_current_medoids, n_samples):
    """
    Find the new clusters given the medoids

    Parameters
    ----------
    distance_matrix: np.array of size (n_samples, n_samples)
        The distance matrix
    indices_of_current_medoids: np.array of size (n_medoids,)
        The indices of the medoids
    n_samples: int
        The value of n_samples

    Returns
    -------
    labels: np.array of size (n_samples,)
        The clusters. Datapoint with index `di` is associated to the medoid with index
        `indices_of_current_medoids[labels[di]]`
    """
    labels = np.empty((n_samples,), dtype=np.int32)
    for dpi in range(n_samples):
        labels[dpi] = np.argmin(np.array([distance_matrix[dpi, mi] for mi in indices_of_current_medoids]))
    return labels


@njit
def nv_matrix_update_medoids(labels, distance_matrix, n_medoids, multiplicity):
    """
    Find the medoids of the clusters given the current labels

    Parameters
    ----------
    labels: np.array of size (n_samples,)
        The label of each datapoint
    distance_matrix: np.array of size(n_samples, n_samples)
        The distance matrix
    n_medoids: int
        The value of n_medoids
    multiplicity: np.array of size (n_samples,)
        The multiplicity of each datapoint

    Returns
    -------
    new_medoids: np.array of size(n_medoids)
        The new medoids, given the labels
    """
    new_medoid_indices = np.empty((n_medoids,), dtype=np.int32)
    for i in range(n_medoids):
        cluster_indices = np.where(labels == i)[0]
        best_index = -1
        best_cost = np.inf

        # new_meds[i] = argmin_{r \in R_i} (sum_{r' in R_i} D(r')*\delta(r',r))
        for r in cluster_indices:

            cost = np.sum(np.array(
                [multiplicity[ri] * distance_matrix[r, ri] for ri in cluster_indices]
            ))

            if cost < best_cost:
                best_cost = cost
                best_index = r

        new_medoid_indices[i] = best_index
    return new_medoid_indices


@njit
def nv_matrix_stop(iteration_number, max_iterations, distance_matrix, indices_of_current_medoids,
                   indices_of_new_medoids, n_medoids, epsilon):
    """
    checks whether to stop k-medoids.

    Parameters
    ----------
    iteration_number: int
        The number of finished iterations
    max_iterations: int
        The maximum number of iterations
    distance_matrix: np.ndarray of size (n_samples, n_samples)
        The distance matrix
    indices_of_current_medoids: np.ndarray of size (n_medoids)
        The indices in the distance matrix with the old medoids
    indices_of_new_medoids: np.ndarray of size (n_medoids)
        The indices in the distance matrix with the new medoids
    n_medoids: int
        The value of n_medoids
    epsilon: float
        The maximum distance between two successive medoids


    Returns
    -------
    ret: bool
        True if all successive medoids are at most `epsilon` apart. False otherwise
    """
    if iteration_number >= max_iterations:
        return True

    for i in range(n_medoids):
        if distance_matrix[indices_of_current_medoids[i], indices_of_new_medoids[i]] > epsilon:
            return False
    return True

# K-medoids++ ----------------------------------------------------------------------------------------------------------
# def kmedoids_plus_plus(distance_matrix, n_medoids, multiplicity=None, initial_medoid_index=0, random_state=0):
#     """
#     k-medoids++ algorithm
#
#     Parameters
#     ----------
#     distance_matrix: np.array of size (n_samples, n_samples)
#         The distance matrix
#     n_medoids: int
#         The desired number of medoids (n_medoids)
#     multiplicity: np.array of size (n_samples,)
#         The multiplicity of the datapoints
#     initial_medoid_index: int
#         Index of the first medoid
#     random_state: int
#         Seed for np.random
#
#     Returns
#     -------
#     medoid_indices: np.array of size (n_medoids,)
#         The initial medoids for clustering.
#     """
#     n_samples = len(distance_matrix)
#     medoid_indices = [initial_medoid_index]
#     non_medoid_indices = list(range(n_samples))
#     non_medoid_indices.remove(initial_medoid_index)
#
#     if multiplicity is None:
#         multiplicity = np.ones(shape=(n_samples,))
#
#     old_state = np.random.get_state()
#
#     np.random.seed(random_state)
#     while len(medoid_indices) < n_medoids:
#         probabilities = np.array([min([multiplicity[ri] * distance_matrix[mi, ri] for mi in medoid_indices])
#                                   for ri in non_medoid_indices])
#         new_medoid_index = np.random.choice(non_medoid_indices, size=1, p=probabilities / sum(probabilities))[0]
#         medoid_indices.append(new_medoid_index)
#         non_medoid_indices.remove(new_medoid_index)
#     np.random.set_state(old_state)
#
#     return np.array(medoid_indices)
#
