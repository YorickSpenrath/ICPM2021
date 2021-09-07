import bitbooster.strings as bbs
import numpy as np
import pandas as pd


def u64(*x):
    """
    Generate a np.array of dtype uint64 from input. Shortcut for np.array(x, dtype=np.uint64)

    Parameters
    ----------
    x: iterable of Number
        Input for the array

    Returns
    -------
    arr: np.ndarray of type np.uint64
        One-dimensional u64 array
    """
    return np.array(x, dtype=np.uint64)


def as_list(x, t):
    """
    Helper function to create a one element list if input is not a list yet. This guarantees that the returned object
    is correctly iterated over; which is convenient to allow single and multi input parameter values.

    Parameters
    ----------
    x: (iterable of) Object
        Object of type t, or an iterable of type t
    t: type
        The type the objects need to be

    Returns
    -------
    as_list: List of t
        List of type t. If the input was a collection; it is simply converted in to a list. If it was of type t, it is
        converted to a one-element list.

    """
    if isinstance(x, t):
        ret = [x]
    else:
        ret = list(x)
    assert all([isinstance(xi, t) for xi in ret])
    return ret


def get_properties_from_labels(labels, min_pts):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    label_counts_non_noise = label_counts[unique_labels != bbs.NOISE_INT]

    res = pd.Series(dtype=float)
    res[bbs.NUMBER_OF_FOUND_CLUSTERS] = len(label_counts_non_noise)
    res[bbs.NOISE_FRACTION] = (labels == bbs.NOISE_INT).mean()
    res[bbs.LARGEST_CLUSTER_SIZE] = label_counts_non_noise.max()
    res[bbs.SMALLEST_CLUSTER_SIZE] = label_counts_non_noise.min()
    res[bbs.MEAN_CLUSTER_SIZE] = label_counts_non_noise.mean()
    res[bbs.STD_CLUSTER_SIZE] = label_counts_non_noise.std()
    res[bbs.NUMBER_OF_CLUSTERS_AT_LEAST_MIN_PTS] = (label_counts_non_noise >= min_pts).sum()

    return res


property_columns = list(get_properties_from_labels(np.arange(-1, 5), 1).index)
