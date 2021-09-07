from bitbooster import strings
from bitbooster.euclidean.bitbooster import EuclideanBinaryObject
from bitbooster.manhattan.bitbooster import ManhattanBinaryObject
from bitbooster.abstract.abdo import AbstractBinaryDataObject
from bitbooster.preprocessing.binarizer import binarize
from bitbooster.preprocessing.discretizers import discretize
from bitbooster.preprocessing.normalizer import normalize


def preprocess(df_in, n_bits, metric, sort_columns=False, weight_dict=None):
    """
    Full preprocessing of data into a binary clusterable class
    Parameters
    ----------
    df_in: pd.Dataframe
        Input data
    n_bits: int
        Number of bits for the resulting BitBooster object
    metric: str
        Metric of the BitBooster object
    sort_columns: bool
        Whether to sort the df_in's columns. If False, the df_in columns are assume to be sorted, if True, the df_in
        columns are sorted
    weight_dict: dict or None
        Weights dictionary passed into transform.

    Returns
    -------
    c: AbstractBinaryDataObject
        The clusterable instance

    """
    bin_data = binarize(discretize(df_in, n_bits, sort_columns=sort_columns, weight_dict=weight_dict), n_bits)
    if metric == strings.EUCLIDEAN:
        return EuclideanBinaryObject(bin_data, num_bits=n_bits, num_features=len(df_in.columns))
    elif metric == strings.MANHATTAN:
        return ManhattanBinaryObject(bin_data, num_bits=n_bits, num_features=len(df_in.columns))
    else:
        raise NotImplementedError(metric)
