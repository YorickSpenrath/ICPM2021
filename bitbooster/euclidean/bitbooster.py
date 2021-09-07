from bitbooster.abstract.abdo import AbstractBinaryDataObject
from bitbooster.euclidean.bitbooster_functions import rectangle_distance_matrix_euclidean_bn2, \
    index_with_lowest_sum_euclidean_bn2


class EuclideanBinaryObject(AbstractBinaryDataObject):

    def __init__(self, data, num_bits, num_features=None, index=None):
        if num_bits > 3:
            raise NotImplementedError('Euclidean not implemented for n>3')
        super().__init__(data, num_bits, num_features, index)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_euclidean_bn2(vertical_data, horizontal_data)

    def _index_with_lowest_sum(self, vertical_data, horizontal_data):
        return index_with_lowest_sum_euclidean_bn2(vertical_data, horizontal_data)
