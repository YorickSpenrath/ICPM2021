import numba
import numpy as np
from numba import njit, u8

# BASE FUNCTIONS -------------------------------------------------------------------------------------------------------
from bitbooster.operations.hamming_weight import hamming_weight


@njit(numba.uint64(numba.uint64[:]))
def extended_or(vec):
    v = 0
    for vi in vec:
        v = v | vi
    return v


@njit(numba.uint64[:](numba.uint64[:]))
def extended_left_shift_1(vec):
    ret = np.zeros(shape=(len(vec),), dtype=np.uint64)
    ret[:-1] = vec[1:]
    return ret


@njit(numba.uint64[:](numba.uint64[:], numba.uint64[:]))
def extended_minus(vec_x, vec_y):
    y = np.uint64(extended_or(vec_y))
    while y != 0:
        vec_b = np.bitwise_and(np.bitwise_not(vec_x), vec_y)
        vec_x = np.bitwise_or(np.bitwise_and(~y, vec_x), np.bitwise_and(y, np.bitwise_xor(vec_x, vec_y)))
        vec_y = extended_left_shift_1(vec_b)
        y = extended_or(vec_y)
    return vec_x


@njit(numba.uint64[:](numba.uint64[:], numba.uint64[:]))
def extended_plus(vec_x, vec_y):
    y = np.uint64(extended_or(vec_y))
    while y != 0:
        vec_b = np.bitwise_and(vec_x, vec_y)
        vec_x = np.bitwise_or(np.bitwise_and(~y, vec_x), np.bitwise_and(y, np.bitwise_xor(vec_x, vec_y)))
        vec_y = extended_left_shift_1(vec_b)
        y = extended_or(vec_y)
    return vec_x


# EUCLIDEAN STUB -------------------------------------------------------------------------------------------------------
@njit(u8[:](u8[:], u8[:]))
def extended_multiply(vec_a, vec_b):
    zero = np.array([0], dtype=np.uint64)
    vec_c = np.zeros_like(vec_a)
    b = extended_or(vec_b)
    while b:
        vec_c = np.concatenate((zero, extended_plus(vec_c, np.bitwise_and(vec_b[-1], vec_a))))
        vec_a = np.concatenate((vec_a, zero))
        vec_b = vec_b[:-1]
        b = extended_or(vec_b)
    return vec_c[1:]


# SHIFT AND SUM --------------------------------------------------------------------------------------------------------
@njit  # (u8(u8[:]))
def shift_and_sum(vec):
    ret = np.uint64(0)
    n = vec.shape[0]
    for i in range(n):
        ret += hamming_weight(vec[i]) << (n - i - 1)
    return ret
