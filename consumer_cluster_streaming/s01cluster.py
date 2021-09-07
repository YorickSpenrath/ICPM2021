import numpy as np
import pandas as pd

from bitbooster import strings as bbs
from bitbooster.utils.objects import discrete_clusterable_from_dataframe
from consumer_cluster_streaming import common as sc
from consumer_cluster_streaming.common import CCSManager, keep_feats, time_name_ty, consumer_description_log
from functions.dataframe_operations import import_sr, import_df, export_df
from functions.progress import ProgressShower


def get_matrices(ccs_manager: CCSManager, t):
    ta = ccs_manager.settings.ta
    tb = ccs_manager.settings.tb
    sr_births = import_sr(ccs_manager.fn_births)
    train_times = [k for k in time_name_ty.keys() if t - tb - ta <= k < t - tb]
    test_times = [k for k in time_name_ty.keys() if t - tb <= k < t]
    all_times = train_times + test_times

    # TODO there are some consumers that do not have any purchases in train_times/all_times
    eligible_consumers = sr_births[sr_births <= t - ccs_manager.settings.ta - ccs_manager.settings.tb].index

    def imp(bn):
        return import_df(consumer_description_log(bn)).set_index(sc.consumer)

    f_matrices = [imp(time_name_ty[k]) for k in all_times]
    all_consumers = set().union(*[set(df.index) for df in f_matrices]).intersection(eligible_consumers)

    # TODO fill_value=0 is not completely right for some
    f_matrices = [df.reindex(all_consumers, fill_value=0) for df in f_matrices]

    return f_matrices, all_consumers, all_times


def get_polyfit(ccs_manager: CCSManager, t):
    f_matrices, all_consumers, all_times = get_matrices(ccs_manager, t)

    features = sorted([k for k in keep_feats if k != sc.NUMBER_OF_WEEKS])

    add_r = len(f_matrices) > 2

    data = np.empty((len(all_consumers), (2 + add_r) * len(features)), dtype=float)

    def mat(feature):
        return np.array([df.loc[:, feature].to_numpy() for df in f_matrices])

    for i, f in enumerate(features):
        (a, b), r, *_ = np.polyfit(np.array(all_times), mat(f), 1, full=True)
        data[:, (2 + add_r) * i + 0] = a
        data[:, (2 + add_r) * i + 1] = b
        if add_r:
            data[:, (2 + add_r) * i + 2] = r

    adapted = sum([[f + '_' + i for i in list('abr' if add_r else 'ab')] for f in features], [])

    return pd.DataFrame(data=data, columns=adapted, index=all_consumers)


def do(ccs_manager: CCSManager):
    """
    Clusters the consumers using the linear fit clustering features

    Parameters
    ----------
    ccs_manager: CCSManager
        Manager to apply on
    """

    # parameters
    for t in ProgressShower(ccs_manager.timestamps, pre=f'clustering [{ccs_manager.name}]'):
        fn_out = ccs_manager.fn_clusters(t)
        if fn_out.exists():
            continue

        if ccs_manager.settings.n_consumer_clusters <= 1:
            # 1 or no clusters

            _, all_consumers, _ = get_matrices(ccs_manager, t)

            if ccs_manager.settings.n_consumer_clusters == 1:
                # Single cluster --> every consumer belongs to the same cluster
                data = 0
            elif ccs_manager.settings.n_consumer_clusters == 0:
                # No clusters --> every consumer gets their own cluster
                data = range(len(all_consumers))
            else:
                raise NotImplementedError()

            sr = pd.Series(index=pd.Index(data=all_consumers, name=sc.consumer),
                           data=data,
                           name=sc.CLUSTER)
        else:
            # Actual clustering
            df_polyfit = get_polyfit(ccs_manager, t)

            c = discrete_clusterable_from_dataframe(
                original_data=df_polyfit,
                n=4, metric=bbs.EUCLIDEAN, weighted=True)

            k = min(ccs_manager.settings.n_consumer_clusters, c.size)

            sr = pd.Series(index=pd.Index(data=df_polyfit.index, name=sc.consumer),
                           data=c.voronoi(k)[1],
                           name=sc.CLUSTER)

        export_df(sr, fn=fn_out)


if __name__ == '__main__':
    do(CCSManager('test'))
