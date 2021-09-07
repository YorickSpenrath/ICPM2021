import numpy as np
from numba import njit, u8, f4

from bitbooster.euclidean.bitbooster_functions import __test
from bitbooster.operations.hamming_weight import hamming_weight
from bitbooster.operations.extended_bit_ops import extended_minus, extended_plus, shift_and_sum


# n=1 ------------------------------------------------------------------------------------------------------------------


@njit(u8[:](u8, u8))
def manhattan_b1_pre_hamming(x0, y0):
    return np.array([x0 ^ y0], dtype=np.uint64)


@njit(f4(u8, u8))
def manhattan_b1(x0, y0):
    return hamming_weight(x0 ^ y0)


# n=2 ------------------------------------------------------------------------------------------------------------------
@njit(u8[:](u8, u8, u8, u8))
def manhattan_b2_pre_hamming(x1, x0, y1, y0):
    return np.array([(x1 ^ y1) & ((x0 ^ y1) | (x1 ^ y0)), x0 ^ y0], dtype=np.uint64)


@njit(f4(u8, u8, u8, u8))
def manhattan_b2(x1, x0, y1, y0):
    return (hamming_weight((x1 ^ y1) & ((x0 ^ y1) | (x1 ^ y0))) << 1) + (hamming_weight(x0 ^ y0))


# n=3 ------------------------------------------------------------------------------------------------------------------
@njit(u8[:](u8, u8, u8, u8, u8, u8))
def manhattan_b3_pre_hamming(x2, x1, x0, y2, y1, y0):
    a = (x2 ^ y2) | (x1 ^ y0) | (x0 ^ y1)
    b = (x1 ^ y1) | ((x0 ^ y0) & (x2 ^ y2))
    c = ~((x2 ^ x0) & (x1 ^ y1) & (x2 ^ y2) & (x0 ^ y0))
    d = (x1 ^ y1) | (x0 ^ x2) | (y0 ^ y2)
    return np.array([
        (x2 ^ y2) & ((x2 ^ y1) | (x1 ^ y2)) & ((x1 ^ y1) | (x2 ^ y0) | (x0 ^ y2)),
        a & b & c & d,
        x0 ^ y0
    ], dtype=np.uint64)


@njit(f4(u8, u8, u8, u8, u8, u8))
def manhattan_b3(x2, x1, x0, y2, y1, y0):
    a = (x2 ^ y2) | (x1 ^ y0) | (x0 ^ y1)
    b = (x1 ^ y1) | ((x0 ^ y0) & (x2 ^ y2))
    c = ~((x2 ^ x0) & (x1 ^ y1) & (x2 ^ y2) & (x0 ^ y0))
    d = (x1 ^ y1) | (x0 ^ x2) | (y0 ^ y2)
    return (hamming_weight((x2 ^ y2) & ((x2 ^ y1) | (x1 ^ y2)) & ((x1 ^ y1) | (x2 ^ y0) | (x0 ^ y2))) << 2) + \
           (hamming_weight(a & b & c & d) << 1) + \
           (hamming_weight(x0 ^ y0))


# n=g ------------------------------------------------------------------------------------------------------------------
@njit(u8[:](u8[:], u8[:]))
def manhattan_bg_pre_hamming(vec_x, vec_y):
    vec_x_signed = np.zeros((len(vec_x) + 1,), dtype=np.uint64)
    vec_x_signed[1:] = vec_x
    vec_y_signed = np.zeros((len(vec_y) + 1,), dtype=np.uint64)
    vec_y_signed[1:] = vec_y
    vec_z = extended_minus(vec_x_signed, vec_y_signed)
    zeros_vec_z = np.zeros((len(vec_x),), dtype=np.uint64)
    zeros_vec_z[-1] = vec_z[0]
    return extended_plus(np.bitwise_xor(vec_z[1:], vec_z[0]), zeros_vec_z)


@njit(f4(u8[:], u8[:]))
def manhattan_bg_vector(vec_x, vec_y):
    return shift_and_sum(manhattan_bg_pre_hamming(vec_x, vec_y))


# Distance Matrix ------------------------------------------------------------------------------------------------------
@njit(f4[:, :](u8[:, :], u8[:, :]))
def rectangle_distance_matrix_manhattan_bn(x_val, y_val):
    len_x = x_val.shape[0]
    len_y = y_val.shape[0]
    n = x_val.shape[1]
    distance_matrix = np.empty((len_x, len_y), dtype=f4)

    if n == 1:
        for i, x0 in enumerate(x_val[:, 0]):
            for j, y0 in enumerate(y_val[:, 0]):
                distance_matrix[i, j] = manhattan_b1(x0, y0)
    elif n == 2:
        for i, (x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1])):
            for j, (y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1])):
                distance_matrix[i, j] = manhattan_b2(x1, x0, y1, y0)
    elif n == 3:
        for i, (x2, x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1], x_val[:, 2])):
            for j, (y2, y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1], y_val[:, 2])):
                distance_matrix[i, j] = manhattan_b3(x2, x1, x0, y2, y1, y0)

    return distance_matrix


@njit(f4[:, :](u8[:, :], u8[:, :]))
def rectangle_distance_matrix_manhattan_bn2(x_val, y_val):
    return __test(x_val, y_val, manhattan_b1, manhattan_b2, manhattan_b3)
