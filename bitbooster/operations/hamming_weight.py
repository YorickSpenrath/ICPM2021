import numpy as np
from numba import vectorize, u8


@vectorize([u8(u8)], nopython=True)
def hamming_weight(x):
    """
    Computes the hamming weight of x.
    Implementation as per https://expobrain.net/2013/07/29/hamming-weights-python-implementation/

    Parameters
    ----------
    x: int
        The value for which to compute the Hamming weight.

    Returns
    -------
        The Hamming weight of x.

    """

    x -= np.bitwise_and(np.right_shift(x, np.uint64(1)), np.uint64(0x5555555555555555))

    x = np.bitwise_and(x, np.uint64(0x3333333333333333)) + \
        np.bitwise_and(np.right_shift(x, np.uint64(2)), np.uint64(0x3333333333333333))

    x += np.right_shift(x, np.uint64(4))
    x = np.bitwise_and(x, np.uint64(0x0f0f0f0f0f0f0f0f))

    # the following line will actually produce an overflow error, but this is fine. The reason is that you *want* to
    # truncate this to 64 bits anyway
    x *= np.uint64(0x0101010101010101)

    # The following line is used in the hamming_weight_old implementation, because python integers are not truncated
    # over 64 bits. It is not needed here, because of the truncation that numpy does apply.
    # x = np.bitwise_and(x, np.uint64(0xffffffffffffffff))

    x = np.right_shift(x, np.uint64(56))
    return x
