def int2bitvector(i, c):
    """
    Get the bitvector representation of an integer.

    Parameters
    ----------
    i: int
        The integer to convert to bitvector
    c: int or None
        The length of the resulting vector. If the bit length of :math:`i` is larger than c, c is ignored. If None, the
        resulting length will be such that there are no leading 0's.

    Returns
    -------
    vec: iterable
        The vector value of the integer

    """
    if c is None:
        return [int(j) for j in f'{i:b}']
    else:
        return [int(j) for j in f'{i:0{c}b}']


def bitvector2int(bitvector):
    """
    Computes the integer representation of a bit vector. The inverse of int2bitvector

    Parameters
    ----------
    bitvector: iterable(int)
        The bit vector

    Returns
    -------
    i: int
        The integer representation of the bit vector

    Raises
    ------
    AssertionError:
        If any of the values in bitvector is not an int in [0, 1]
    """
    assert all([isinstance(i, int) for i in bitvector])
    assert all([i in [0, 1] for i in bitvector])
    return int(''.join([str(i) for i in bitvector]), 2)


def subset2bitvector(subset, superset_list):
    """
    Computes the bitvector representation of a subset.

    Parameters
    ----------
    subset: iterable
        The subset.
    superset_list: list
        The superset.

    Returns
    -------
    v: list(int)
        The bitvector representation of the subset based on the superset.
    """
    assert isinstance(superset_list, list)
    return [(1 if s in subset else 0) for s in superset_list]


def subset2int(subset, superset_list):
    """
    Computes the integer representation of a subset.

    Parameters
    ----------
    subset: iterable
        The subset.
    superset_list: list
        The superset.

    Returns
    -------
    i: int
        The integer representation of the subset based on the superset.

    Raises
    ------
    AssertionError
        If the list is not sorted.
    """
    assert sorted(superset_list) == superset_list, 'The superset list should be sorted. ' \
                                                   'You might have forgotten this somewhere else too'
    return bitvector2int(subset2bitvector(subset, superset_list))


def ints2bitvectors(ints):
    """
    Converts multiple integers to same-length bitvectors

    Parameters
    ----------
    ints: iterable of int
        The integers

    Returns
    -------
    bit_vectors: iterable of list of ints
        The bit_vector representation of the integers in ints, all with the same length.
    """
    assert all([isinstance(i, int) for i in ints])
    bit_length = len(f'{max(ints):b}')
    return [int2bitvector(i, bit_length) for i in ints]


def bit_square(a):
    if a == 0:  # 0^2 = 0
        return 0
    if a == 1:  # 1^2 = 1
        return 1
    if a & 1:  # a^2 = (2n+1)^2 = 4n^2 + 4n + 1 = 4 floor(a/2) + 2 (a-1) + 1
        return (bit_square(a >> 1) << 2) + ((a ^ 1) << 1) + 1
    return bit_square(a >> 1) << 2  # a^2 = (2n)^2 = 4n^2 = 4 floor(a/2)
