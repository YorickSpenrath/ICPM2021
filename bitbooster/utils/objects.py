import bitbooster.strings as bbs


def bitbooster_from_dataframe(df, n, metric, weighted):
    from bitbooster.preprocessing import binarize, discretize
    data = binarize(discretize(df, n_bits=n), n)
    if metric == bbs.EUCLIDEAN:
        if weighted:
            from bitbooster.euclidean.weighted import WeightedEuclideanBitBooster as Cl
        else:
            from bitbooster.euclidean.bitbooster import EuclideanBinaryObject as Cl
        return Cl(data=data, num_bits=n)
    else:
        raise NotImplementedError(f'Not implemented for metric={metric}, weighted={weighted}')


def vanilla_clusterable_from_dataframe(original_data, metric, weighted, normalize_data=True):
    if normalize_data:
        from bitbooster.preprocessing.normalizer import normalize
        new_data = normalize(original_data)
    else:
        new_data = original_data

    if metric == bbs.EUCLIDEAN:
        if weighted:
            from bitbooster.euclidean.weighted import VanillaWeightedEuclidean as Cl
        else:
            from bitbooster.euclidean.vanilla import EuclideanVanillaObject as Cl
        return Cl(data=new_data)
    else:
        raise NotImplementedError(f'Not implemented for metric={metric}, weighted={weighted}')


def discrete_clusterable_from_dataframe(original_data, n, metric, weighted):
    if n <= 3:
        return bitbooster_from_dataframe(original_data, n, metric, weighted)
    else:
        from bitbooster.preprocessing import binarize, discretize
        discretized_data = discretize(original_data, n)
        return vanilla_clusterable_from_dataframe(discretized_data, metric, weighted, False)
