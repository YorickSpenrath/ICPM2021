import pandas as pd
import numpy as np
from numba import njit, u8, int_


def binarize(in_df, n):
    assert isinstance(in_df, pd.DataFrame), 'df must be a DataFrame'
    assert isinstance(n, int), 'n must be an integer'
    assert n > 0, 'n must be positive'
    assert all([in_df[c].dtype in (int, np.uint64, np.int64) for c in in_df.columns]), 'Values must be int or (u)int64'
    assert sorted(in_df.columns) == list(in_df.columns), 'Columns of the DataFrame must be sorted'
    # noinspection PyTypeChecker,PyUnresolvedReferences
    assert ((0 <= in_df) & (in_df < 2 ** n)).all().all(), 'Values must be in [0, 2**n)'

    # TODO you can probably win something here too for frames with fewer than 33 columns
    #  but I think the Hamming weight breaks down then
    # noinspection PyTypeChecker
    assert len(in_df.columns) <= 64, 'This is not implemented for more than 64 columns'
    out_df = pd.DataFrame(data=_compute_binarized(in_df.to_numpy(dtype=np.uint64), n), index=in_df.index,
                          columns=range(n))
    return out_df[sorted(out_df.columns, reverse=True)]


@njit(u8[:](u8[:, :], u8[:]))
def _dot(mat, arr):
    n, k = mat.shape

    res = np.zeros(shape=(n,), dtype=np.uint64)
    for j in range(k):
        res = res + mat[:, j] * arr[j]

    return res


@njit(u8[:](u8[:, :], int_, u8[:]))
def _compute_single(in_array, significant_bit, pow_array):
    z = u8(np.power(2, significant_bit))

    l_and = np.bitwise_and(in_array, z)
    floor_div = np.floor_divide(l_and, z)
    dotted = _dot(floor_div, pow_array)
    return dotted


@njit(u8[:, :](u8[:, :], int_))
def _compute_binarized(in_array, n_bits):
    pow_array = np.empty(shape=(in_array.shape[1]), dtype=np.uint64)
    for k in range(in_array.shape[1]):
        pow_array[-k - 1] = u8(np.power(2, k))
    out_array = np.zeros(shape=(in_array.shape[0], n_bits), dtype=np.uint64)
    for i in range(n_bits):
        out_array[:, i] = _compute_single(in_array, i, pow_array)
    return out_array
