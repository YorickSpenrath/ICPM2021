import numpy as np
import functools

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error, balanced_accuracy_score, \
    recall_score, precision_score, f1_score

from consumer_cluster_streaming import common as sc
from functions.dataframe_operations import import_df, import_sr, export_df
from functions.progress import ProgressShower
from consumer_cluster_streaming.common import get_xy_from_css, get_dfs_from_range, get_y, CCSManager

top_k_fraction = 0.1
top_k_fraction_string = '0.1'


def classification_thing(df_turnover):
    df_turnover = df_turnover.replace({0: pd.NA})

    a = ~df_turnover.prev.isna()
    b = ~df_turnover.this.isna()
    c = ~df_turnover.pred.isna()

    abs_true_labels = pd.Series(index=df_turnover.index, dtype=float)
    rel_true_labels = pd.Series(index=df_turnover.index, dtype=float)
    abs_pred_labels = pd.Series(index=df_turnover.index, dtype=float)
    rel_pred_labels = pd.Series(index=df_turnover.index, dtype=float)

    def get_abs_true(mask):
        return df_turnover.prev[mask] - df_turnover.this[mask]

    def get_abs_pred(mask):
        return df_turnover.prev[mask] - df_turnover.pred[mask]

    m = a & b & c
    # All good
    abs_true_labels[m] = get_abs_true(m)
    abs_pred_labels[m] = get_abs_pred(m)
    rel_true_labels[m] = get_abs_true(m).divide(df_turnover.prev[m])
    rel_pred_labels[m] = get_abs_pred(m).divide(df_turnover.prev[m])

    m = ~a & b & c
    # No previous purchase
    abs_true_labels[m] = -df_turnover.this[m]
    abs_pred_labels[m] = -df_turnover.pred[m]
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    m = a & ~b & c
    # No purchase in last week
    abs_true_labels[m] = df_turnover.prev[m]
    abs_pred_labels[m] = get_abs_pred(m)
    rel_true_labels[m] = 1
    rel_pred_labels[m] = get_abs_pred(m).divide(df_turnover.prev[m])

    m = a & b & ~c
    # Previous but no predicted purchase, this should not be possible
    abs_true_labels[m] = get_abs_true(m)
    abs_pred_labels[m] = df_turnover.prev[m]
    rel_true_labels[m] = get_abs_true(m).divide(df_turnover.prev[m])
    rel_pred_labels[m] = 1

    m = a & ~b & ~c
    # Previous but no predicted purchase, this should not be possible
    abs_true_labels[m] = df_turnover.prev[m]
    abs_pred_labels[m] = df_turnover.prev[m]
    rel_true_labels[m] = 1
    rel_pred_labels[m] = 1

    m = ~a & b & ~c
    # New purchase, not predicted. These consumers are not important
    abs_true_labels[m] = -np.inf
    abs_pred_labels[m] = -np.inf
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    m = ~a & ~b & c
    # Prediction, but it is not relevant
    abs_true_labels[m] = -np.inf
    abs_pred_labels[m] = -np.inf
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    m = ~a & ~b & ~c
    # These consumers never purchased anything, and were not predicted to do so
    abs_true_labels[m] = -np.inf
    abs_pred_labels[m] = -np.inf
    rel_true_labels[m] = -np.inf
    rel_pred_labels[m] = -np.inf

    # Verify we have everything
    for sr in [abs_true_labels, abs_pred_labels, rel_true_labels, rel_pred_labels]:
        assert sr.isna().sum() == 0

    def label(sr_):
        srx_ = sr_.sort_index()
        srx_ = srx_.sort_values(kind='mergesort', ascending=False)
        srx_.iloc[:int(top_k_fraction * len(sr_))] = True
        srx_.iloc[int(top_k_fraction * len(sr_)):] = False
        return srx_.astype(bool).sort_index()

    abs_true_labels = label(abs_true_labels)
    abs_pred_labels = label(abs_pred_labels)
    rel_true_labels = label(rel_true_labels)
    rel_pred_labels = label(rel_pred_labels)

    metric_names = ['accuracy', 'recall', 'precision', 'f1']
    metric_methods = [balanced_accuracy_score, recall_score, precision_score, f1_score]

    res = pd.Series(dtype=float)

    for mn, m in zip(metric_names, metric_methods):
        res[f'bottom_absolute_{mn}'] = m(abs_true_labels, abs_pred_labels)
        res[f'bottom_relative_{mn}'] = m(rel_true_labels, rel_pred_labels)

    return res


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def amape(y_true, y_pred):
    y_pred = y_pred[y_true != 0]
    y_true = y_true[y_true != 0]
    return mean_absolute_percentage_error(y_true, y_pred)


def get_consumer_turnover(t0, te):
    def red(x, y):
        return x[sc.TOTAL_VALUE].add(y[sc.TOTAL_VALUE], fill_value=0)

    z = get_dfs_from_range(t0=t0, te=te)

    return functools.reduce(red, z, pd.DataFrame(columns=[sc.TOTAL_VALUE]))


def combine_predictions(ccs_manager: CCSManager):
    if all(map(lambda fn: fn.exists(), [ccs_manager.fn_all_cluster_predictions,
                                        ccs_manager.fn_all_consumer_predictions,
                                        ccs_manager.fn_all_turnover_predictions])):
        return

    turnover_df = pd.DataFrame()

    # Cluster predictions
    cluster_predictions = pd.DataFrame()

    # Consumer predictions
    consumer_predictions = pd.DataFrame()

    for t in ProgressShower(ccs_manager.timestamps[:-1], pre=f'extracting predictions [{ccs_manager.name}]'):

        # Predictions for each cluster
        clusters = import_sr(ccs_manager.fn_clusters(t))
        cluster_pred = import_df(ccs_manager.fn_cluster_predictions(t)).set_index(sc.CLUSTER)['y_pred']
        cluster_prev = pd.Series(get_xy_from_css(ccs_manager, t, t)[1])
        cluster_sizes = clusters.to_frame().groupby('cluster').size()
        cluster_factor = cluster_pred.divide(cluster_prev)
        # factor is corrected if cluster_prev was 0. This is fine, because the individual consumers also have 0
        cluster_factor[cluster_prev == 0] = 0

        # Predictions for each consumer
        consumer_prev = get_consumer_turnover(t - ccs_manager.settings.tb, t)
        consumer_factor = clusters.replace(cluster_factor.to_dict())

        # Correct for missing ids in clusters / previous
        all_consumer_ids = {*set(clusters.index), *set(consumer_prev.index)}
        factor_star = cluster_pred.multiply(cluster_sizes).sum() / cluster_prev.multiply(cluster_sizes).sum()

        # Consumers without previous visits have 0 turnover
        consumer_prev = consumer_prev.reindex(all_consumer_ids, fill_value=0)
        # Consumer without cluster are assigned the default
        consumer_factor = consumer_factor.reindex(all_consumer_ids, fill_value=factor_star)
        consumer_pred = consumer_prev.multiply(consumer_factor)
        for cluster in cluster_prev[cluster_prev == 0].index:
            consumer_pred[clusters[clusters == cluster].index] = cluster_pred[cluster]

        consumer_predictions[t] = consumer_pred
        cluster_predictions[t] = cluster_pred
        turnover_df.loc[t, sc.CLUSTER] = cluster_pred.multiply(cluster_sizes).sum()

    consumer_predictions.index.name = sc.consumer
    cluster_predictions.index.name = sc.CLUSTER
    export_df(consumer_predictions, ccs_manager.fn_all_consumer_predictions, index=True)
    export_df(cluster_predictions, ccs_manager.fn_all_cluster_predictions, index=True)

    turnover_df[sc.consumer] = consumer_predictions.fillna(0).sum(axis=0)
    turnover_df.index.name = 't'
    export_df(turnover_df, ccs_manager.fn_all_turnover_predictions, index=True)


def combine_ground_truths(ccs_manager: CCSManager):
    fn_consumer = ccs_manager.fn_all_consumer_ground_truths
    fn_cluster = ccs_manager.fn_all_cluster_ground_truths
    if all(map(lambda x: x.exists(), [fn_consumer, fn_cluster, ccs_manager.fn_all_turnover_ground_truths])):
        return

    if fn_consumer.exists():
        df_consumer = import_df(fn_consumer).set_index(sc.consumer)
        df_consumer.columns = map(int, df_consumer.columns)
    else:
        df_consumer = pd.DataFrame()

    if fn_cluster.exists():
        df_cluster = import_df(fn_cluster).set_index(sc.CLUSTER)
        df_cluster.columns = map(int, df_cluster.columns)
    else:
        df_cluster = pd.DataFrame()

    for t in ProgressShower(ccs_manager.timestamps, pre=f'extracting ground truth [{ccs_manager.name}]'):
        if t not in df_consumer.columns:
            sr_t = get_consumer_turnover(t0=t, te=t + ccs_manager.settings.tb)
            sr_t.name = t
            df_consumer = df_consumer.join(other=sr_t, how='outer').fillna(0)
        if t not in df_cluster.columns:
            sr_t = get_y(t0=t, te=t + ccs_manager.settings.tb, clusters=import_sr(ccs_manager.fn_clusters(t)))
            sr_t = pd.Series(sr_t, name=t)
            df_cluster = df_cluster.join(other=sr_t, how='outer').fillna(0)

    df_consumer.index.name = sc.consumer
    df_cluster.index.name = sc.CLUSTER

    export_df(df_consumer[sorted(df_consumer.columns)], fn_consumer, index=True)
    sr_turnover = df_consumer.sum(axis=0)
    sr_turnover.index.name = 't'
    sr_turnover.name = 'turnover'
    export_df(sr_turnover, ccs_manager.fn_all_turnover_ground_truths, index=True)
    export_df(df_cluster[sorted(df_cluster.columns)], fn_cluster, index=True)


def save_results_df(ccs_manager: CCSManager):
    if ccs_manager.fn_results.exists():
        return
    df = pd.DataFrame(columns=ccs_manager.timestamps[:-1])

    df_consumers_pred = import_df(ccs_manager.fn_all_consumer_predictions).set_index(sc.consumer)
    df_consumers_true = import_df(ccs_manager.fn_all_consumer_ground_truths).set_index(sc.consumer)

    df_clusters_pred = import_df(ccs_manager.fn_all_cluster_predictions).set_index(sc.CLUSTER)
    df_clusters_true = import_df(ccs_manager.fn_all_cluster_ground_truths).set_index(sc.CLUSTER)

    for dfx in [df_consumers_pred, df_consumers_true, df_clusters_pred, df_clusters_true]:
        dfx.columns = map(int, dfx.columns)

    df_turnover_pred = import_df(ccs_manager.fn_all_turnover_predictions).set_index('t')
    df_turnover_true = import_sr(ccs_manager.fn_all_turnover_ground_truths)

    for t in ProgressShower(ccs_manager.timestamps[:-1], pre=f'Extracting errors [{ccs_manager.name}]'):
        gtt = df_turnover_true[t]
        df.loc['turnover_error_cluster', t] = (df_turnover_pred.loc[t, sc.CLUSTER] - gtt) / gtt
        df.loc['turnover_error_consumer', t] = (df_turnover_pred.loc[t, sc.consumer] - gtt) / gtt

        for agg_name in ['consumer', 'cluster', 'consumerNEW']:
            if agg_name.startswith('consumer'):
                y_true = df_consumers_true[t].fillna(0)
                y_pred = df_consumers_pred[t]
                if agg_name == 'consumer':
                    y_pred = y_pred.dropna()
                    y_true = y_true.reindex(y_pred.index, fill_value=0)
                else:
                    y_pred = y_pred.reindex(y_true.index).fillna(0)
            else:
                y_true = df_clusters_true[t]
                y_pred = df_clusters_pred[t]
                if ccs_manager.settings.n_consumer_clusters == 0:
                    y_true = y_true.iloc[:len(y_pred)]

            for metric_name, metric in zip(['r2', 'rmse', 'mape', 'mape0'],
                                           [r2_score, rmse, mean_absolute_percentage_error, amape]):
                df.loc[f'{metric_name}_{agg_name}', t] = metric(y_true=y_true, y_pred=y_pred)

        # Consumer drop ------------------------------------------------------------------------------------------------

        df_turnover = pd.DataFrame()
        df_turnover['prev'] = get_consumer_turnover(t - 1, t)
        df_turnover['this'] = get_consumer_turnover(t, t + 1)
        df_turnover['pred'] = df_consumers_pred[t]
        res = classification_thing(df_turnover)
        for k, v in res.to_dict().items():
            df.loc[f'{k}_consumer', t] = v

        df_turnover['pred'] = import_sr(ccs_manager.fn_clusters(t)).replace(df_clusters_pred[t].to_dict())
        res = classification_thing(df_turnover)
        for k, v in res.to_dict().items():
            df.loc[f'{k}_cluster', t] = v

    df.index.name = 'metric'
    export_df(df, ccs_manager.fn_results, index=True)


def redo(ccs_manager: CCSManager):
    if ccs_manager.fn_results.exists():
        ccs_manager.fn_results.unlink()
    do(ccs_manager)


def do(ccs_manager):
    combine_ground_truths(ccs_manager)
    combine_predictions(ccs_manager)
    save_results_df(ccs_manager)
