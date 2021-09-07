import pandas as pd
from abc import ABC

import numpy as np

from bitbooster.abstract.clusterable import Clusterable


class WeightedClusterable(Clusterable, ABC):

    def __init__(self, data, index=None, column_names=None):
        Clusterable.__init__(self, data, index=index, column_names=column_names)

        df = self.dataframe
        self._original_index_dtype = df.index.dtype

        # self._reverse_map[i] are all indices in the original df that are mapped to index i
        self._reverse_map = [list(v.index) for k, v in df.groupby(list(df.columns), as_index=False)]

        repeat_labels = sum([[i for _ in v] for i, v in enumerate(self._reverse_map)], [])

        # self._index_map[j] is the index in the new clusterable to which the index j in the original df is mapped
        self._index_map = pd.Series(data=repeat_labels, index=sum(self._reverse_map, [])).reindex(df.index).to_numpy()

        df = df.groupby(list(df.columns), as_index=False).size()
        self._data = df.iloc[:, :-1].to_numpy()
        self._names = df.index.to_numpy()
        self._weights = df.iloc[:, -1].to_numpy().astype(np.int32)

    @property
    def unique_size(self):
        # guaranteed to be unique
        return self.size

    @property
    def actual_size(self):
        return sum(self._weights)

    def get_index_with_lowest_sum(self, index0, index1):
        # TODO: you can speed this up with partial sums if the dataset becomes too large
        #  however; the key idea of a Weighted Clusterable is that your dataset is small anyway.
        m = self.get_sub_distance_matrix(index0, index1)
        wm = (self._weights[index0] * m.T).T
        wms = wm.sum(axis=0)
        return np.argmin(wms)

    def k_medoids_plus_plus(self, k, seed=0):
        self._k_medoids_plus_plus_check(k, seed)

        rng = np.random.RandomState(seed)
        imi = [rng.choice(self._index_map)]

        sdm = self.get_sub_distance_matrix(None, [imi[0]]).flatten()
        while len(imi) < k:
            i = rng.choice(self._index_map, p=sdm[self._index_map] / sum(sdm[self._index_map]))
            sdm_p = self.get_sub_distance_matrix(None, [i]).flatten()
            sdm = np.fmin(sdm, sdm_p)
            imi.append(i)

        return imi

    def voronoi_out(self, fn_out, labels, new_medoid_indices, out_sep):
        raise Exception('This is not valid for weighted class')

    def voronoi_from_imi(self, initial_medoid_indices, **kwargs):
        res = super().voronoi_from_imi(initial_medoid_indices, **kwargs)
        new_medoids = np.array([self._reverse_map[i][0] for i in res[0]], dtype=self._original_index_dtype)
        new_labels = res[1][self._index_map]
        return (new_medoids, new_labels) + res[2:]

    def dbscan_cluster(self, eps, min_points, return_stats=False):
        raise Exception('This is not valid for weighted class')
        # res = super().dbscan_cluster(eps=eps, min_points=min_points, return_stats=return_stats)
        # new_labels = self._convert_labels(res[0])
        # return (new_labels,) + res[1:]

    def get_dist(self, min_pts=4, frac=1, random_state=0):
        raise Exception('This is not valid for weighted class')

    def distance_matrix_to_other(self, other, index_self, index_other):
        raise Exception('This is not valid for weighted class')
