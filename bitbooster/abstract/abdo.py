from abc import ABC

import pandas as pd
import numpy as np

from bitbooster.abstract.clusterable import Clusterable


class AbstractBinaryDataObject(Clusterable, ABC):

    def __init__(self, data, num_bits, num_features=None, index=None):
        """
        Default initialization for BinaryDataObject

        Parameters
        ----------
        data: pd.DataFrame or np.array
            The data
        num_bits: int
            Number of bits. Must be positive
        num_features: int or None
            Number of original features. Must be positive
        index: iterable of str or None
            Names of the datapoints. If None, data must be of type pd.DataFrame; names will be taken from its index.
            If not None and data is of type pd.DataFrame, index will be ignored and a warning is raised.

        Raises
        ------
        Assertion Error:
            If shape of the data does not match names and number of bits.
            If maximum value in the data exceeds 2 ** num_features.
            If minimum value in the data is below 0.

        Notes
        -----
        Data is converted to numpy array in super class. This class assumes that the FIRST column is the most
        significant bit. I.e. if the data consists of datapoints x,y represented in 2 bits, we have that:

        data =
              | column 0     | column 1
        row 0 | x_bin^{2,1} | x_bin^{2,0}
        row 0 | y_bin^{2,1} | y_bin^{2,0}

        Though this might seem counter-intuitive; it is closest to the definitions in the paper, as well as closest
        to bit-representation; where the left/first index represents the most significant bit.

        """

        # Bit-wise values, index
        super().__init__(data, index)

        # Number of bits used in encoding
        assert isinstance(num_bits, int)
        assert num_bits > 0
        self.n = num_bits

        # Number of features encoded
        if num_features is not None:
            assert isinstance(num_features, int)
            assert num_features > 0

        assert self._data.dtype in [np.uint64, int]
        self._data = np.array(self._data, dtype=np.uint64)

        # Check if data values match shape and range
        assert self._data.shape[1] == self.n
        assert (self._data < 0).sum(axis=None) == 0

        if num_features is not None:
            # noinspection PyUnresolvedReferences
            assert (self._data >= (2 ** num_features)).sum(axis=None) == 0

    def drop_duplicates(self, **kwargs):
        return type(self)(data=self.dataframe.drop_duplicates(**kwargs), num_bits=self.n)
