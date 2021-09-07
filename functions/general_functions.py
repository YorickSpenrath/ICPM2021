"""
Useful functions that do not belong to any other file
"""
import datetime


def assert_positive_int(i, name=None):
    assert isinstance(i, int) and i > 0, '' if name is None else f'{name} must be a positive int'


def assert_positive_int_or_none(i, name=None):
    assert (i is None) or (isinstance(i, int) and i > 0), \
        '' if name is None else f'{name} must be a positive int or None'


def padded_enumerated_string(pre, num):
    """
    Create a list of strings, such that each has the form 'pre_i' where i is padded such that each string has the same
    length; and such that its lexicographic sorting is identical to the sorting of the integer values of i.

    Parameters
    ----------
    pre: str
        first part of the enumerated text
    num: iterable or int
        Number of strings to generate. Must be >=1. If not int, the len of num is used instead

    Returns
    -------
    pes: list of str
        List of strings as described above.
    """
    # Verify input
    if not isinstance(num, int):
        num = len(num)
    assert num >= 1

    # Padding
    x = len(str(num - 1))

    # Return value
    return [f'{pre}_{i:0{x}}' for i in range(num)]


def listified(values, t, validation=None, filtering=None, sort=False):
    """
    Transform single ``t`` or collection of ``t`` to a List of ``t``, with several extra options

    Parameters
    ----------
    values : t or iterable of t
        The values to listify
    t : Type or iterable of Type
        The required type of the values. If iterable of t, all types are accepted, and all values are converted to the
        first of these. See notes for exceptions
    validation: Callable ``t`` -> Boolean or iterable or None
        Each of the values needs to match this condition. Alternatively, if an iterable is given, each of the value must
        be in this iterable.
    filtering: Callable ``t`` -> Boolean or iterable or None
        Values are filtered out if they do not match this condition. Alternatively, if an iterable is given, only values
        in this iterable are kept.
    sort
        Whether to sort the resulting list before returning. Values are sorted ascending.

    Returns
    -------
    values : List
        A list of type ``t`` or ``t[0]``, with only the values that match the ``filtering`` condition. Sorted if
        ``sort`` is True. Validated for given ``validation``

    Raises
    ------
    AssertionError:
        If a value is not of (any) correct type
        If a value is not callable, and t == callable
        If a filtered value does not meet the ``validation`` condition
        If ``t`` is an iterable, but not all values of t are of type type

    Notes
    -----
    If t is the function "callable", then all values will be checked whether they are callable
    """
    if t is callable:
        # callable: all values must be callable
        if callable(values):
            values = [values]
        else:
            values = list(values)
            for v in values:
                assert callable(v), f'value {v} is not callable'
    elif isinstance(t, type):
        # single type, all must conform
        if isinstance(values, t):
            values = [values]
        else:
            values = list(values)
            for v in values:
                assert isinstance(v, t), f'value {v} is not of type {t}'
    else:
        # multiple types
        t = listified(t, type)
        if any([isinstance(values, ti) for ti in t]):
            values = [t[0](values)]
        else:
            values = list(values)
            for v in values:
                assert any([isinstance(v, ti) for ti in t]), f'value {v} is none of types {t}'
            values = [t[0](v) for v in values]

    if filtering is not None:
        if callable(filtering):
            values = [v for v in values if filtering(v)]
        else:
            values = [v for v in values if v in filtering]

    if validation is not None:
        if callable(validation):
            for v in values:
                assert validation(v), f'value [{v}] does not match given validation'
        else:
            for v in values:
                assert v in validation, f'value [{v}] is not in given validation'

    if sort:
        values = sorted(values)

    return values


def get_time_str(add_ms=False):
    """
    Get current time as string

    Parameters
    ----------
    add_ms: bool
        Add the current ms to the string

    Returns
    -------
    time_str: str
        The current time as string, formatted as "%Y-%m-%d_%H%M%S" or "%Y-%m-%d_%H%M%S.%f"
    """
    f = "%Y-%m-%d_%H%M%S"
    if add_ms:
        f += ".%f"
    return datetime.datetime.now().strftime(f)


def assert_valid_partition(full_set, partition):
    """
    Assert that the given iterable of sets is a partition of the entire set

    Parameters
    ----------
    full_set: set
        The complete set
    partition: iterable of set
        The partition

    Raises
    ------
    AssertionError
        - If the partition contains an empty set
        - The union of the partitions do not recreate the original set
        - There are elements that are in multiple partition sets

    """
    partition = [set(pi) for pi in partition]
    full_set = set(full_set)

    foo = []
    for p in partition:
        # Each set in the partition is non-empty
        assert len(partition) > 0, 'Partition contains an empty set'

        foo.extend(p)

    assert set(foo) == full_set, 'Union of the partition is not the original set'
    assert len(foo) == len(set(foo)), 'Duplicate elements in the partition'


def large_number_formatting(v, decimals=0, __post_fix=''):
    """
    Formats large numbers (1000 -> K), (1e6 -> M), (1e9 -> G).

    Parameters
    ----------
    v : float
        The value
    decimals : int
        number of decimals in the final answer
    __post_fix : str
        what to add after the results (for recursive computations of numbers of at least than 1e12)

    Returns
    -------
    fmt: str
        K/M/G string representation of the large number
    """
    if v < 1e3:
        return f'{v:.{decimals}f}{__post_fix}'
    for border, prefix in zip([1e6, 1e9, 1e12], 'KMG'):
        if v < border:
            return large_number_formatting(v / border * 1e3, decimals, prefix + __post_fix)
    return large_number_formatting(v / 1e12, decimals, 'T' + __post_fix)
