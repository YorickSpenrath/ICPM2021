import warnings
from abc import abstractmethod, ABC

import numpy as np
import pandas as pd
from numba import njit, u4, f4, b1

from bitbooster.abstract.errors import NotEnoughUniqueDatapointsException, KMedoidsNotEnoughLabelsException
from bitbooster.strings import UNDEFINED_INT, NOISE_INT, INNER_NEIGHBOURHOOD_COMPUTATIONS, \
    OUTER_NEIGHBOURHOOD_COMPUTATIONS


class Clusterable(ABC):

    def __init__(self, data, index=None, column_names=None):

        # Convert data
        if isinstance(data, pd.DataFrame):
            if index is not None:
                warnings.warn('Variable "index" is ignored if data is of type DataFrame')
            if column_names is not None:
                warnings.warn('Variable "column_names" is ignored if data is of type DataFrame')
            index = np.array([str(s) for s in data.index])
            column_names = np.array([str(c) for c in data.columns])
            data = data.to_numpy()
        elif isinstance(data, np.ndarray):
            if index is not None:
                index = np.array(index, dtype=str)
                assert data.shape[0] == index.shape[0]
            if column_names is not None:
                column_names = np.array(column_names, dtype=str)
                assert data.shape[1] == column_names.shape[0]
        else:
            raise ValueError(f'Unknown data type : f{type(data)}')

        assert data.ndim == 2
        self._data = data
        self._names = index
        self._column_names = column_names

    # PROPERTIES -------------------------------------------------------------------------------------------------------
    @property
    def names(self):
        if self._names is None:
            return np.arange(self._data.shape[0])
        return self._names

    @property
    def data(self):
        return self._data

    @property
    def column_names(self):
        if self._column_names is None:
            return np.arange(self._data.shape[1])
        return self._column_names

    @property
    def size(self):
        return self._data.shape[0]

    @property
    def unique_size(self):
        return np.unique(self._data, axis=0).shape[0]

    # noinspection SpellCheckingInspection
    @property
    def dataframe(self):
        # noinspection PyTypeChecker
        return pd.DataFrame(data=self._data, index=pd.Index(self.names, name='name'), columns=self.column_names)

    # METHODS ----------------------------------------------------------------------------------------------------------
    # Matrix
    def _subdata(self, index):
        """
        Retrieve a subset of the data, given in indices

        Parameters
        ----------
        index: iterable of int or None
            If None, all data is returned. Otherwise, only the data with given indices

        Returns
        -------

        """
        if index is None:
            return self._data
        else:
            return self._data[index, :]

    def get_sub_distance_matrix(self, index0, index1):
        """
        Get a sub-matrix of the distance matrix, based on the given indices

        Parameters
        ----------
        index0: iterable of int or None
            Indices for the vertical data (i.e. rows of the distance matrix)
        index1: iterable of int or None
            Indices for the horizontal data (i.e. columns of the distance matrix)
        Returns
        -------
        dm: np.array of size (len(index0), len(index1)) of dtype float32
            Sub-distance matrix
        """
        d0 = self._subdata(index0)
        d1 = self._subdata(index1)
        return self._rectangle_distance_matrix(d0, d1)

    def get_index_with_lowest_sum(self, index0, index1):
        """
        Computes the index from index0 which has the lowest total distance to index1

        Parameters
        ----------
        index0: iterable of int or None
            Indices for the vertical data (i.e. rows of the distance matrix)
        index1
            Indices for the horizontal data (i.e. rows of the distance matrix)

        Returns
        -------
        i: int
            Index in index0 such that sum(d(i,j), j in index1) is minimal for i in index0

        """
        d0 = self._subdata(index0)
        d1 = self._subdata(index1)
        return self._index_with_lowest_sum(d0, d1)

    def _k_medoids_plus_plus_check(self, k, seed):
        assert isinstance(k, int), 'k should be integer'
        assert 0 < k, 'k should be positive'
        if k > self.unique_size:
            raise NotEnoughUniqueDatapointsException(self.unique_size, k)
        assert isinstance(seed, int)

    # Clustering
    def k_medoids_plus_plus(self, k, seed=0):
        """
        k_medoids++ algorithm. Returns indices.

        Parameters
        ----------
        k: int
            number of initial medoid indices
        seed: int
            random state for numpy

        Returns
        -------
        imi: iterable of int
            initial medoid indices found using k_medoids++
        """
        self._k_medoids_plus_plus_check(k, seed)

        # save and set random state
        state = np.random.get_state()
        np.random.seed(seed)

        # Selected random first index
        imi = [np.random.randint(0, self.size)]

        # Probabilities for next
        sdm = self.get_sub_distance_matrix(None, [imi[0]]).flatten()
        while len(imi) < k:
            i = np.random.choice(range(self.size), p=sdm / sum(sdm))
            sdm_p = self.get_sub_distance_matrix(None, [i]).flatten()
            sdm = np.fmin(sdm, sdm_p)
            imi.append(i)

        # Reset random state
        # noinspection PyTypeChecker
        np.random.set_state(state)

        return imi

    def voronoi_out(self, fn_out, labels, new_medoid_indices, out_sep):
        sr = pd.Series(labels)
        sr.name = 'medoid index'
        sr.index.name = 'datapoint index'

        if self._names is not None:
            sr.index = self._names
            sr = sr.map({lab: n for lab, n in enumerate(self._names[new_medoid_indices])})
            sr.index.name = 'datapoint name'
            sr.name = 'medoid name'
        sr.to_csv(fn_out, sep=out_sep)

    def voronoi_from_imi(self, initial_medoid_indices, **kwargs):
        """
        Parameters
        ----------
        initial_medoid_indices

        Other Parameters
        ----------------
        max_iter: int
            Maximum number of iterations. Default is 1000_000
        eps: float
            Maximum distance between two subsequent medoids. Default is 0.001
        fn_out: str or Path or None. Default is None
            If not None, result is written to this location. Each row contains two values, separated by `out_sep`. The
            first value of these is the name (or index) of a datapoint, the second value is the name (or index) of the
            medoid of the cluster to which the datapoint belongs.
        out_sep: str. Default is ";"
            If fn_out is not None, this is used as separator. If fn_out is None, this is ignored
        return_iteration_count: bool. Default is False
            If True, the number of iterations is returned.
        Returns
        -------
        mi: iterable of int
            Indices of final medoids
        labels: iterable of int
            Labels of final medoids

        """
        # Optional, additional parameters
        max_iter = kwargs.pop('max_iter', 1000_000)
        eps = kwargs.pop('eps', 0.001)
        fn_out = kwargs.pop('fn_out', None)
        out_sep = kwargs.pop('out_sep', ';')
        return_iteration_count = kwargs.pop('return_iteration_count', False)

        if kwargs:
            raise ValueError(f'Unknown setting for cluster_from_imi:{kwargs}')

        old_medoid_indices = initial_medoid_indices.copy()
        n_medoids = len(initial_medoid_indices)
        n_samples = self._data.shape[0]

        iteration_number = 0

        while True:
            iteration_number += 1
            labels = nv_metric_update_clusters(matrix=self.get_sub_distance_matrix(None, old_medoid_indices),
                                               n_samples=n_samples)
            if len(np.unique(labels)) != n_medoids:
                raise KMedoidsNotEnoughLabelsException(n_clusters=len(initial_medoid_indices),
                                                       n_labels=len(np.unique(labels)))

            new_medoid_indices = np.empty(n_medoids, dtype=np.int32)
            for i in range(n_medoids):
                label_indices = np.where(labels == i)[0]
                new_medoid_indices[i] = label_indices[self.get_index_with_lowest_sum(label_indices, label_indices)]

            if nv_metric_stop(iteration_number=iteration_number,
                              max_iter=max_iter,
                              matrix=self.get_sub_distance_matrix(old_medoid_indices, new_medoid_indices),
                              eps=eps,
                              n_medoids=n_medoids):
                break
            else:
                old_medoid_indices = new_medoid_indices

        if fn_out is not None:
            self.voronoi_out(fn_out, labels, new_medoid_indices, out_sep)

        if return_iteration_count:
            return new_medoid_indices, labels, iteration_number
        else:
            return new_medoid_indices, labels

    def cluster(self, k=None, initial_medoid_indices=None, initial_medoid_names=None, **kwargs):
        warnings.warn('Warning: method "cluster" is deprecated as there are multiple clustering algorithms '
                      'implemented. Please use "voronoi" for the previous definition of "cluster"',
                      category=DeprecationWarning)
        return self.voronoi(k, initial_medoid_indices, initial_medoid_names, **kwargs)

    def voronoi(self, k=None, initial_medoid_indices=None, initial_medoid_names=None, **kwargs):
        """
        Cluster the data object
        # TODO initial_medoid_names takes priority over others

        Parameters
        ----------
        k: int or None
            Number of clusters to find. Must be positive
        initial_medoid_indices: iterable of int or None
            Indices of initial medoids
        initial_medoid_names: iterable of str
            Names of the initial medoids

        Other Parameters
        ----------------
        See :py:meth:`cluster_from_imi()
            <bitbooster.abstract.clusterable.Clusterable.cluster_from_imi>`.

        Returns
        -------
        See :py:meth:`cluster_from_imi()
            <bitbooster.abstract.clusterable.Clusterable.cluster_from_imi>`.

        Notes
        -----
        The clustering depends on the values for k and initial_medoid_indices:
        - If k is None, len(initial_medoid_indices) cluster are found, starting from initial_medoid_indices
        - If initial_medoid_indices is None, k clusters are found, starting with k-medoids++
        - If neither are None, a verification is done that k == len(initial_medoid_indices), after which k is ignored
        - Initializing form k-medoids++ using a given seed is currently not supported

        Raises
        ------
        ValueError
           If k and initial_medoid_indices are both None
        """
        if initial_medoid_names is not None:
            assert set(initial_medoid_names).issubset(self.names)
            initial_medoid_indices = np.where(np.isin(self.names, initial_medoid_names))[0]
        else:
            if k is None and initial_medoid_indices is None:
                raise ValueError('Must give k or initial_medoid_indices')
            if k is None and initial_medoid_indices is not None:
                pass
            if k is not None and initial_medoid_indices is None:
                initial_medoid_indices = self.k_medoids_plus_plus(k)
            if k is not None and initial_medoid_indices is not None:
                assert k == len(initial_medoid_indices), 'k and initial_medoid_indices are given, but they do not match'

        return self.voronoi_from_imi(initial_medoid_indices, **kwargs)

    def dbscan_cluster(self, eps, min_points, return_stats=False):
        """
        Does a DBSCAN cluster, as defined in https://doi.org/10.1145%2F3068335, except without the trees

        Parameters
        ----------
        eps: float
            See paper
        min_points: int
            See paper
        return_stats: bool
            If True, returns a dict with s. n additional tuple (a,b), with a the number of points that where checked to
            be core points in the outer loop (Line 3 of Algorithm 1), and b the number of points that where checked to
            be core points in the inner loop (Line 13 of Algorithm 1).
        Returns
        -------
        lab: np.ndarray of int
            Labels, where -1 are outliers
        """
        inner = 0
        outer = 0

        cluster_idx = 0
        labels = np.ones(self.size, dtype=int) * UNDEFINED_INT
        for p in range(self.size):
            if labels[p] != UNDEFINED_INT:
                continue
            outer += 1
            neighbors = np.where(self.get_sub_distance_matrix([p], None)[0] < eps)[0]
            if len(neighbors) < min_points:
                labels[p] = NOISE_INT
                continue
            cluster_idx += 1
            labels[p] = cluster_idx
            seed_set = set(neighbors).difference({p})
            while seed_set:
                q = seed_set.pop()
                if labels[q] == NOISE_INT:
                    labels[q] = cluster_idx
                if labels[q] != UNDEFINED_INT:
                    continue
                labels[q] = cluster_idx
                inner += 1
                neighbors = np.where(self.get_sub_distance_matrix([q], None)[0] < eps)[0]
                if len(neighbors) >= min_points:
                    seed_set.update(neighbors[labels[neighbors] < 0])

        if return_stats:
            return labels, {OUTER_NEIGHBOURHOOD_COMPUTATIONS: outer, INNER_NEIGHBOURHOOD_COMPUTATIONS: inner}
        else:
            return labels

    def get_dist(self, min_pts=4, frac=1, random_state=0):

        # Decide which indices to use
        if frac == 1:  # No need to sample
            indices = range(self.size)
            num_samples = self.size
        else:
            # Need to sample
            # Save and set random state
            state = np.random.get_state()
            np.random.seed(random_state)
            # Compute number of samples used
            if frac < 1:
                num_samples = np.floor(self.size * frac)
            else:
                num_samples = int(frac)
            # noinspection PyTypeChecker
            np.random.set_state(state)

            # Get sample indices
            indices = np.random.choice(range(self.size), np.floor(self.size * frac), replace=False)

        # Compute 4dist for the (sampled) indices
        dist = np.empty(num_samples, dtype=np.float32)
        for i in range(0, num_samples, 1_000):
            min_idx = i
            max_idx = min(i + 1_000, num_samples)
            dm = self.get_sub_distance_matrix(indices[min_idx: max_idx], None)
            dist[min_idx:max_idx] = np.partition(dm, min_pts, axis=1)[:, min_pts]
        return dist

    def distance_matrix_to_other(self, other, index_self, index_other):
        """
        Compute the distance matrix to another clusterable. This clusterable is on the row index, other is on the
        column index.

        Parameters
        ----------
        other: Clusterable
            Other clusterable. Must be of the same type.
        index_self: iterable of int or None
            Indices of self to use. If None, all are used.
        index_other: iterable of int or None
            Indices of other to use. If None, all are used.

        Returns
        -------
        dm: np.ndarray
            Distance matrix, such that dm[i,j] is the distance between self[index_self[i]] and other[index_other[j]],
            or self[i] / other[j] if the respective index argument is None.
        """
        assert isinstance(other, type(self))
        d_self = self._subdata(index_self)
        d_other = other._subdata(index_other)
        return self._rectangle_distance_matrix(d_self, d_other)

    # ABSTRACT METHODS -------------------------------------------------------------------------------------------------

    @abstractmethod
    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        pass

    @abstractmethod
    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        pass


@njit(u4[:](f4[:, :], u4))
def nv_metric_update_clusters(matrix, n_samples):
    ret = np.empty(n_samples, dtype=np.uint32)
    for i in range(n_samples):
        ret[i] = np.argmin(matrix[i, :])
    return ret


@njit(u4(f4[:, :], u4))
def nv_compute_index_of_medoid(matrix, n_samples):
    s = np.empty(n_samples)
    for i in range(n_samples):
        s[i] = np.sum(matrix[i, :])
    return np.argmin(s)


@njit(b1(u4, u4, f4[:, :], f4, u4))
def nv_metric_stop(iteration_number, max_iter, matrix, eps, n_medoids):
    if iteration_number >= max_iter:
        return True

    for i in range(n_medoids):
        if matrix[i, i] > eps:
            return False

    return True


def get_dimensions(data):
    if isinstance(data, np.ndarray):
        return data.shape
    elif isinstance(data, pd.DataFrame):
        return len(data), len(data.columns)
    else:
        raise TypeError(f'Unable to get dimensions for type {type(data)}')
