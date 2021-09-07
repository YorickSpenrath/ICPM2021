from unittest.case import TestCase
import numpy as np
from numpy.testing import assert_array_equal

from bitbooster.operations.bit_ops import int2bitvector, subset2int, subset2bitvector, ints2bitvectors, bitvector2int, \
    bit_square
from bitbooster.operations.hamming_weight import hamming_weight
from bitbooster.operations.extended_bit_ops import extended_or, extended_left_shift_1, extended_minus, extended_plus, \
    extended_multiply, shift_and_sum
from bitbooster.common import u64


class TestExtendedBitOps(TestCase):

    def test_extended_or1(self):
        x = u64(3, 2, 1, 8, 11)
        y = 11
        self.assertEqual(extended_or(x), y)

    def test_extended_or2(self):
        x = u64(3, 2, 1, 8)
        y = 11
        self.assertEqual(extended_or(x), y)

    def test_extended_or3(self):
        x = u64(1, 2, 4, 8, 16, 32, 64, 128, 256)
        y = 511
        self.assertEqual(extended_or(x), y)

    def test_extended_left_shift_1(self):
        x = u64(3, 4, 525, 45)
        z = u64(4, 525, 45, 0)
        assert_array_equal(extended_left_shift_1(x), z)

    def test_extended_minus(self):
        x = u64(2, 3, 3)
        y = u64(2, 1, 2)
        z = u64(0, 2, 1)
        assert_array_equal(extended_minus(x, y), z)

    def test_extended_plus(self):
        x = u64(1, 2, 3)
        y = u64(0, 2, 0)
        z = u64(3, 0, 3)
        assert_array_equal(extended_plus(x, y), z)

    # def test_extended_difference1(self):
    #     x = u64(3, 2, 3)
    #     y = u64(3, 1, 3)
    #     z = u64(0, 3, 0)
    #     assert_array_equal(extended_difference(x, y), z)

    # def test_extended_difference2(self):
    #     x = u64(120, 778, 671, 702)
    #     y = u64(927, 126, 633, 341)
    #     z = u64(5, 402, 548, 1003)
    #     assert_array_equal(extended_difference(x, y), z)

    def test_extended_multiply(self):
        # 3 * 3 = 9
        # 8 * 8 = 64
        # 4 * 4 = 16
        # 6 * 6 = 36
        x = u64(4, 3, 9, 8)
        z = u64(4, 1, 2, 8, 1, 0, 8)
        assert_array_equal(extended_multiply(x, x), z)

    def test_extended_multiply2(self):
        # 8 * 8 = 64
        x = u64(1, 0, 0, 0)
        z = u64(1, 0, 0, 0, 0, 0, 0)
        assert_array_equal(extended_multiply(x, x), z)


class TestBitOps(TestCase):
    def test_int2bitvector(self):
        a = [1, 0, 1, 0, 1]
        k = 21
        self.assertListEqual(a, int2bitvector(k, c=None))

    def test_subset2int(self):
        superset = [1, 2, 3, 4, 5]
        subset = [1, 3, 5]
        self.assertEqual(subset2int(subset, superset), 21)

    def test_subset2bitvector(self):
        superset = [1, 2, 3, 4, 5]
        subset = [1, 3, 5]
        self.assertListEqual(subset2bitvector(subset, superset), [1, 0, 1, 0, 1])

    def test_ints2bitvectors(self):
        k = [21, 9]
        a = [[1, 0, 1, 0, 1], [0, 1, 0, 0, 1]]
        self.assertListEqual(ints2bitvectors(k), a)

    def test_bitvector2int(self):
        a = [1, 0, 1, 0, 1]
        k = 21
        self.assertEqual(bitvector2int(a), k)

    def test_hamming_weight(self):
        self.assertEqual(hamming_weight(np.uint64(21)), 3)

    def test_bit_square(self):
        for i in range(100):
            self.assertEqual(bit_square(i), i * i)

    def test_shift_and_sum(self):
        # 1 0 1 0 1
        self.assertEqual(shift_and_sum(u64(1, 0, 1, 0, 1)), 21)

    def test_shift_and_sum2(self):
        # 1 0 1 0 1 --> 21
        # 1 1 1 0 1 --> 29
        # 0 0 0 1 1 -->  3
        # ----------------
        # 6 2 6 1 7 --> 53
        self.assertEqual(shift_and_sum(u64(6, 2, 6, 1, 7)), 53)
