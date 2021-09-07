import warnings

import numpy as np
import pandas as pd


def hotfix_transform(df, n_bits, sort_columns=False, weight_dict=None):
    warnings.warn('This function has been renamed to discretize, which can be imported from preprocess directly')
    discretize(df, n_bits, sort_columns, weight_dict)


def discretize(df, n_bits, sort_columns=False, weight_dict=None):
    """
    Transforms data to discretized bitbooster values

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to be converted
    n_bits: int
        Number of bits to use
    sort_columns: bool
        Whether to explicitly sort the columns
    weight_dict: dict or None
        If not None, a custom scale for each feature

    Returns
    -------
    df_transformed: pd.DataFrame
        Transformed DataFrame
    """
    assert isinstance(df, pd.DataFrame)
    from sklearn.preprocessing import MinMaxScaler

    # Scale to [0, 2^n]
    mms = MinMaxScaler(feature_range=(0, 2 ** n_bits))
    mms.fit(df.to_numpy())
    new_data = mms.transform(df)

    if weight_dict is not None:
        assert set(df.columns) == set(weight_dict.keys()), 'weight dict given but does not match columns'
        assert all([0 <= w <= 2 ** n_bits for w in weight_dict.values()]), 'weight dict contains illegal values'
        assert all([np.issubdtype(type(w), np.integer) for w in weight_dict.values()]), \
            'weight should only contains integers'
        for i, k in enumerate(df.columns):
            new_data[:, i] = new_data[:, i] * weight_dict[k] / 2 ** n_bits

    # the formula is part of Eq 3, the upper clipping is necessary because transform may result in values like
    # 2.000000004 for n_bits=1, the lower clipping is necessary to make the lowest value in the data part of the 0th bin
    new_data = np.clip(np.ceil(new_data) - 1, 0, 2 ** n_bits - 1).astype(int)
    # Clipping
    df = pd.DataFrame(data=new_data, columns=df.columns, index=df.index)
    if sort_columns:
        return df[sorted(df.columns)]
    else:
        return df
