import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal

from bitbooster.abstract.test import BitBoosterBaseTest
from bitbooster.common import u64
from bitbooster.euclidean.bitbooster import EuclideanBinaryObject
from bitbooster.euclidean.bitbooster_functions import *
from bitbooster.preprocessing.binarizer import binarize


class Test(BitBoosterBaseTest):

    def test_euclidean_b3_1(self):
        # Input
        a = u64(4, 14, 13)
        b = u64(1, 9, 5)

        # Individual terms
        d2 = euclidean_b3_pre_hamming_sqrt(*a, *b)
        assert_array_equal(d2, [5, 0, 0, 7, 0, 8])

        # Individual
        d = euclidean_b3(*a, *b)
        self.assertAlmostEqual(d, 77 ** 0.5, places=6)

    def test_euclidean_b2_1(self):
        # Input
        a = u64(396, 124)
        b = u64(227, 344)

        # Individual
        d2 = euclidean_b2_pre_hamming_sqrt(*a, *b)
        assert_array_equal(d2, [4, 75, 0, 292])

        # Individual
        d = euclidean_b2(*a, *b)
        self.assertAlmostEqual(d, 27 ** 0.5, places=6)

    def test_euclidean_b3_2(self):
        # Input
        a = u64(1144652, 545739, 37257)
        b = u64(2082187, 224855, 1990029)

        # Individual terms
        d2 = euclidean_b3_pre_hamming_sqrt(*a, *b)
        assert_array_equal(d2, [135168, 393795, 822272, 78232, 0, 2018308])

        # Individual input
        d = euclidean_b3(*a, *b)
        self.assertAlmostEqual(d, 237 ** 0.5, places=6)

    def test_square_distance_matrix_euclidean_b3(self):
        a = u64(97, 86, 74)
        b = u64(62, 87, 27)
        c = u64(61, 77, 77)

        dm_true = np.array([[0, 99, 99], [99, 0, 80], [99, 80, 0]], dtype=np.uint64) ** 0.5

        # Individual Computation
        dm_comp = rectangle_distance_matrix_euclidean_bn(x_val=np.array([a, b, c]), y_val=np.array([a, b, c]))
        assert_array_almost_equal(dm_comp, dm_true)

    def full_b1(self):
        self._full(1, euclidean_b1, euclidean_b1_pre_hamming_sqrt)

    def full_b2(self):
        self._full(2, euclidean_b2, euclidean_b2_pre_hamming_sqrt)

    def full_b3(self):
        self._full(3, euclidean_b3, euclidean_b3_pre_hamming_sqrt)

    def test_ebc_1(self):
        a = u64(1144652, 545739, 37257)
        b = u64(2082187, 224855, 1990029)
        df = pd.DataFrame(columns=['x2', 'x1', 'x0'], dtype=np.uint64)
        df.loc[0] = a
        df.loc[1] = b
        res = EuclideanBinaryObject(data=df, num_bits=3).get_sub_distance_matrix([0], None)
        assert_array_almost_equal(res, [[0, 237 ** 0.5]])

    def test_panic(self):
        a = u64(3, 7, 2, 1)
        b = u64(2, 1, 0, 7)

        df = pd.DataFrame(columns=['f1', 'f2', 'f3', 'f4'], dtype=np.uint64)
        df.loc[0] = a
        df.loc[1] = b

        df = binarize(in_df=df, n=3)
        res = EuclideanBinaryObject(data=df, num_bits=3).get_sub_distance_matrix([0], None)
        assert_array_almost_equal(res, [[0, 77 ** 0.5]])
