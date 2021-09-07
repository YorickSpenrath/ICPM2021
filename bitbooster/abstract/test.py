from itertools import product
from unittest import TestCase
import numpy as np

from numpy.testing import assert_array_equal


class BitBoosterBaseTest(TestCase):

    def _full(self, n_bits, d_function, pre_hamming_function):
        powers_of_2 = [2 ** y for y in list(range(n_bits))[::-1]]
        for i, j in product(range(2 ** n_bits), range(2 ** n_bits)):
            vec = [(i & x) // x for x in powers_of_2] + [(j & x) // x for x in powers_of_2]

            d = d_function(*vec)
            d_true = abs(i - j)
            self.assertEqual(d_true, d)

            d2 = pre_hamming_function(*vec)
            d2_true = np.array([(d_true & x) // x for x in powers_of_2], dtype=np.uint64)
            assert_array_equal(d2, d2_true)
