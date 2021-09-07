from consumer_cluster_streaming import s01cluster, s02models, s03predictions, s04results
from consumer_cluster_streaming.common import CCSManager, default_n_consumer_clusters
from consumer_cluster_streaming.s99_combined_results import combine_results

n = default_n_consumer_clusters

redo_results = False


def __x(txr, ncr):
    retrain = True
    for tx in txr:
        for n_clusters in ncr:
            name = f'{"no_" if not retrain else ""}retrain_{tx}.{n_clusters}'

            ccs = CCSManager(name, incremental_training=retrain, ta=tx, n_consumer_clusters=n_clusters)
            if not redo_results and ccs.fn_results.exists():
                continue
            s01cluster.do(ccs)
            s02models.do(ccs)
            s03predictions.do(ccs)
            if redo_results:
                s04results.redo(ccs)
            else:
                s04results.do(ccs)

            if not ccs.cluster_ok:
                break


def run():
    combine_results()

    # Few clusters
    number_of_consumer_clusters = [n // 10] + list(range(n // 4, n + 1, n // 4))
    __x(range(2, 11), number_of_consumer_clusters)

    # Special case: single cluster
    number_of_consumer_clusters = [1]
    __x(range(2, 11), number_of_consumer_clusters)

    # Lots of clusters
    for k in range(2 * n, 10 * n + 1, n):
        number_of_consumer_clusters = list([k])
        __x(range(2, 11), number_of_consumer_clusters)

    # Special case: no cluster
    number_of_consumer_clusters = [0]
    __x(range(2, 11), number_of_consumer_clusters)


if __name__ == '__main__':
    run()
