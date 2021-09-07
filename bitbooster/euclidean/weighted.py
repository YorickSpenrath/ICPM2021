from bitbooster.abstract.weighted_clusterable import WeightedClusterable
from bitbooster.euclidean.bitbooster import EuclideanBinaryObject
from bitbooster.euclidean.vanilla import EuclideanVanillaObject


class WeightedEuclideanBitBooster(WeightedClusterable, EuclideanBinaryObject):

    def __init__(self, data, num_bits, num_features=None, index=None):
        EuclideanBinaryObject.__init__(self, data=data, num_bits=num_bits, num_features=num_features, index=index)
        WeightedClusterable.__init__(self, data=data, index=index, column_names=None)


class VanillaWeightedEuclidean(WeightedClusterable, EuclideanVanillaObject):
    pass

