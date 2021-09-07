from bitbooster.abstract.abdo import AbstractBinaryDataObject
from bitbooster.manhattan.bitbooster_functions import rectangle_distance_matrix_manhattan_bn


class ManhattanBinaryObject(AbstractBinaryDataObject):

    def __init__(self, data, num_bits, num_features=None, index=None):
        if num_bits > 3:
            raise NotImplementedError('Manhattan not implemented for n>3')
        super().__init__(data, num_bits, num_features, index)

    def _rectangle_distance_matrix(self, vertical_data, horizontal_data):
        return rectangle_distance_matrix_manhattan_bn(vertical_data, horizontal_data)
