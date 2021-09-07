from numpy.testing import assert_array_equal

from bitbooster.abstract.test import BitBoosterBaseTest
from bitbooster.manhattan.bitbooster_functions import *


class Test(BitBoosterBaseTest):

    def test_manhattan_b1(self):
        # d(1,0) = 1
        # d(0,1) = 1
        # d(1,0) = 1
        # d(0,0) = 0
        # d(1,1) = 0

        # Distance
        d = manhattan_b1(21, 9)
        self.assertEqual(d, 3)

        # Individual terms
        d2 = manhattan_b1_pre_hamming(21, 9)
        assert_array_equal(d2, np.array([0b11100], dtype=np.uint64))

    def test_manhattan_b2_1(self):
        # d(2,1) = 1 -> 10 01
        # d(1,2) = 1 -> 01 10
        # d(2,1) = 1 -> 10 01
        # d(0,0) = 0 -> 00 00
        # d(3,3) = 0 -> 11 11

        # 10101 -> 21
        # 01001 ->  9
        # 01001 ->  9
        # 10101 -> 21

        # 1+1+1+0+0 = 3
        # Distance
        d = manhattan_b2(21, 9, 9, 21)
        self.assertEqual(d, 3)

        # Individual Terms
        d2 = manhattan_b2_pre_hamming(21, 9, 9, 21)
        assert_array_equal(d2, np.array([0, 0b11100], dtype=np.uint64))

    def test_manhattan_b2_2(self):
        # d(2,2) = 0 -> 10 10 = 00
        # d(2,1) = 1 -> 10 01 = 01
        # d(0,2) = 2 -> 00 10 = 10
        # d(3,0) = 3 -> 11 00 = 11

        # 1101 -> 13
        # 0001 ->  1
        # 1010 -> 10
        # 0100 ->  4

        # 0+1+2+3 = 6
        # Distance
        d = manhattan_b2(13, 1, 10, 4)
        self.assertEqual(d, 6)

        # Individual terms
        d2 = manhattan_b2_pre_hamming(13, 1, 10, 4)
        d2_true = np.array([0b0011, 0b0101], dtype=np.uint64)
        assert_array_equal(d2, d2_true)

    def test_manhattan_b3_1(self):
        a = (823250, 407806, 1519070)
        b = (491646, 1082574, 1454509)

        d = manhattan_b3(*a, *b)
        self.assertEqual(d, 57)

        d2 = manhattan_b3_pre_hamming(*a, *b)
        d2_true = np.array([527244, 1551408, 72819], dtype=np.uint64)
        assert_array_equal(d2_true, d2)

    def test_manhattan_b1_full(self):
        self._full(1, manhattan_b1, manhattan_b1_pre_hamming)

    def test_manhattan_b2_full(self):
        self._full(2, manhattan_b2, manhattan_b2_pre_hamming)

    def test_manhattan_b3_full(self):
        self._full(3, manhattan_b3, manhattan_b3_pre_hamming)

    def test_square_distance_matrix_manhattan_b2(self):
        # a a2  b b2  c
        # 0 00  3 11  3
        # 1 01  2 10  1
        # 2 10  1 01  1
        # 3 11  3 11  0
        # B(a) =  3, 5
        # B(b) = 13,11
        # sum c = 5
        data = np.array([[3, 5], [13, 11]], dtype=np.uint64)
        dm_comp = rectangle_distance_matrix_manhattan_bn(data, data)
        dm_true = np.array([[0, 5], [5, 0]])
        assert_array_equal(dm_comp, dm_true)
