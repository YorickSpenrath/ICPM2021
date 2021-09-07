import numpy as np
from numba import njit, f4

from bitbooster.abstract.vanilla import BaseVanilla


class ManhattanVanillaObject(BaseVanilla):

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_manhattan_vanilla(vertical_data, horizontal_data)


@njit(['f4[:,:](f8[:,:],f8[:,:])', 'f4[:,:](i8[:,:],i8[:,:])', 'f4[:,:](f4[:,:],f4[:,:])', 'f4[:,:](i4[:,:],i4[:,:])'])
def rectangle_distance_matrix_manhattan_vanilla(vertical_data, horizontal_data):
    number_v = vertical_data.shape[0]
    number_h = horizontal_data.shape[0]
    distance_matrix = np.empty((number_v, number_h), dtype=f4)
    for i, vec_i in enumerate(vertical_data):
        for j, vec_j in enumerate(horizontal_data):
            distance_matrix[i, j] = np.sum(np.abs(vec_i - vec_j))
    return distance_matrix
