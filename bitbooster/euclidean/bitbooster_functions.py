import numpy as np
from numba import njit, u8, f4, i8

from bitbooster.operations.hamming_weight import hamming_weight


# n=1 ------------------------------------------------------------------------------------------------------------------
@njit(u8[:](u8, u8))
def euclidean_b1_pre_hamming_sqrt(x0, y0):
    return np.array([x0 ^ y0], dtype=np.uint64)


@njit(f4(u8, u8))
def euclidean_b1(x0, y0):
    return np.sqrt(
        hamming_weight(x0 ^ y0)
    )


# n=2 ------------------------------------------------------------------------------------------------------------------
@njit(u8[:](u8, u8, u8, u8))
def euclidean_b2_pre_hamming_sqrt(x1, x0, y1, y0):
    # m terms
    m00 = x0 ^ y0
    m11 = x1 ^ y1
    return np.array([
        m00 & m11 & ~(x1 ^ x0),
        ~m00 & m11,
        0,
        m00
    ], dtype=np.uint64)


@njit(f4(u8, u8, u8, u8))
def euclidean_b2(x1, x0, y1, y0):
    # m terms
    m00 = x0 ^ y0
    m11 = x1 ^ y1
    return np.sqrt(
        (hamming_weight(m00 & m11 & ~(x1 ^ x0)) << 3) +
        (hamming_weight(~m00 & m11) << 2) +
        hamming_weight(m00)
    )


# n=3 ------------------------------------------------------------------------------------------------------------------
@njit(u8[:](u8, u8, u8, u8, u8, u8))
def euclidean_b3_pre_hamming_sqrt(x2, x1, x0, y2, y1, y0):
    # m terms
    m00 = x0 ^ y0
    m11 = x1 ^ y1
    m22 = x2 ^ y2

    return np.array([
        (x2 & x1 & ~y2 & ~y1 & (x0 | ~y0)) | (y2 & y1 & ~x2 & ~x1 & (y0 | ~x0)),
        m22 & (~x2 | x1 | ~y1) & (~y2 | y1 | ~x1) & (~m22 | ~m11 | m00) & ((x2 ^ y0) | (x0 ^ y2) | m11),
        (m22 & ((x0 & ~y0 & (y1 | ~x1)) | (y0 & ~x0 & (x1 | ~y1)))) | (~m22 & m11 & m00 & ~(x1 ^ x0)),
        (~m00) & m11,
        0,
        m00
    ], dtype=np.uint64)


@njit(f4(u8, u8, u8, u8, u8, u8))
def euclidean_b3(x2, x1, x0, y2, y1, y0):
    # m terms
    m00 = x0 ^ y0
    m11 = x1 ^ y1
    m22 = x2 ^ y2

    return np.sqrt(
        (hamming_weight((x2 & x1 & ~y2 & ~y1 & (x0 | ~y0)) | (y2 & y1 & ~x2 & ~x1 & (y0 | ~x0))) << 5) +
        (hamming_weight(
            m22 & (~x2 | x1 | ~y1) & (~y2 | y1 | ~x1) & (~m22 | ~m11 | m00) & ((x2 ^ y0) | (x0 ^ y2) | m11)) << 4) +
        (hamming_weight(
            (m22 & ((x0 & ~y0 & (y1 | ~x1)) | (y0 & ~x0 & (x1 | ~y1)))) | (~m22 & m11 & m00 & ~(x1 ^ x0))) << 3) +
        (hamming_weight((~m00) & m11) << 2) +
        (hamming_weight(m00))
    )


# Combined -------------------------------------------------------------------------------------------------------------
@njit(f4[:, :](u8[:, :], u8[:, :]))
def rectangle_distance_matrix_euclidean_bn(x_val, y_val):
    len_x = x_val.shape[0]
    len_y = y_val.shape[0]
    n = x_val.shape[1]
    distance_matrix = np.empty((len_x, len_y), dtype=f4)

    if n == 1:
        for i, x0 in enumerate(x_val[:, 0]):
            for j, y0 in enumerate(y_val[:, 0]):
                distance_matrix[i, j] = euclidean_b1(x0, y0)
    elif n == 2:
        for i, (x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1])):
            for j, (y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1])):
                distance_matrix[i, j] = euclidean_b2(x1, x0, y1, y0)
    elif n == 3:
        for i, (x2, x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1], x_val[:, 2])):
            for j, (y2, y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1], y_val[:, 2])):
                distance_matrix[i, j] = euclidean_b3(x2, x1, x0, y2, y1, y0)

    return distance_matrix


@njit
def __test(x_val, y_val, b1f, b2f, b3f):
    len_x = x_val.shape[0]
    len_y = y_val.shape[0]
    n = x_val.shape[1]
    distance_matrix = np.empty((len_x, len_y), dtype=f4)

    if n == 1:
        for i, x0 in enumerate(x_val[:, 0]):
            for j, y0 in enumerate(y_val[:, 0]):
                distance_matrix[i, j] = b1f(x0, y0)
    elif n == 2:
        for i, (x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1])):
            for j, (y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1])):
                distance_matrix[i, j] = b2f(x1, x0, y1, y0)
    elif n == 3:
        for i, (x2, x1, x0) in enumerate(zip(x_val[:, 0], x_val[:, 1], x_val[:, 2])):
            for j, (y2, y1, y0) in enumerate(zip(y_val[:, 0], y_val[:, 1], y_val[:, 2])):
                distance_matrix[i, j] = b3f(x2, x1, x0, y2, y1, y0)

    return distance_matrix


@njit
def __index_with_lowest_sum(x_val, y_val, b1f, b2f, b3f):
    n_vertical = x_val.shape[0]
    n_horizontal = y_val.shape[0]

    lowest_sum = np.inf
    lowest_index = -1

    for i in range(n_vertical):
        i_sum = 0
        x = 0

        while i_sum < lowest_sum and x < n_horizontal:
            i_sum += __test(x_val[i: i + 1], y_val[x: x + 1000], b1f, b2f, b3f).sum()
            x += 1000

        if i_sum < lowest_sum:
            lowest_sum = i_sum
            lowest_index = i

    return lowest_index


@njit(f4[:, :](u8[:, :], u8[:, :]))
def rectangle_distance_matrix_euclidean_bn2(x_val, y_val):
    return __test(x_val, y_val, euclidean_b1, euclidean_b2, euclidean_b3)


@njit(i8(u8[:, :], u8[:, :]))
def index_with_lowest_sum_euclidean_bn2(x_val, y_val):
    return __index_with_lowest_sum(x_val, y_val, euclidean_b1, euclidean_b2, euclidean_b3)
