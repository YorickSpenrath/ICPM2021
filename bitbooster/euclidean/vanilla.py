import numpy as np
from numba import njit, f4

from bitbooster.abstract.vanilla import BaseVanilla


class EuclideanVanillaObject(BaseVanilla):

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return euclidean_index_with_lowest_sum(vertical_data, horizontal_data)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_euclidean_vanilla(vertical_data, horizontal_data)


@njit(['f4[:,:](f8[:,:],f8[:,:])', 'f4[:,:](i8[:,:],i8[:,:])', 'f4[:,:](f4[:,:],f4[:,:])', 'f4[:,:](i4[:,:],i4[:,:])'])
def rectangle_distance_matrix_euclidean_vanilla(vertical_data, horizontal_data):
    number_v = vertical_data.shape[0]
    number_h = horizontal_data.shape[0]
    distance_matrix = np.empty((number_v, number_h), dtype=f4)
    for i, vec_i in enumerate(vertical_data):
        for j, vec_j in enumerate(horizontal_data):
            distance_matrix[i, j] = np.sqrt(np.sum(np.square(vec_i - vec_j)))
    return distance_matrix


@njit(['i4(f8[:,:],f8[:,:])', 'i4(i8[:,:],i8[:,:])', 'i4(f4[:,:],f4[:,:])', 'i4(i4[:,:],i4[:,:])'])
def euclidean_index_with_lowest_sum(x_val, y_val):
    n_vertical = x_val.shape[0]
    n_horizontal = y_val.shape[0]

    lowest_sum = np.inf
    lowest_index = -1

    for i in range(n_vertical):
        i_sum = 0
        x = 0

        while i_sum < lowest_sum and x < n_horizontal:
            i_sum += rectangle_distance_matrix_euclidean_vanilla(x_val[i: i + 1], y_val[x: x + 1000]).sum()
            x += 1000

        if i_sum < lowest_sum:
            lowest_sum = i_sum
            lowest_index = i

    return lowest_index
