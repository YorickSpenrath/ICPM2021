from pathlib import Path

import numpy as np

from functions import file_functions
from functions.dataframe_operations import import_df, import_sr
from functions.file_functions import list_files
from functions.alm import AbstractSettings, AbstractLocationsManager

root = Path('.')
cdl_root = root / 'cdl'
fn_birth_ty = root / 'fn_births.csv'
experiments_root = root / 'experiments'
if cdl_root.exists():
    time_name_ty = {i: bn[:-4] for i, bn in enumerate(sorted(file_functions.list_files(cdl_root, False)))}
else:
    time_name_ty = dict()
keep_feats = ['freshness', 'mean_item_value', 'product_density', 'count', 'number_of_weeks', 'total_item_count',
              'total_value']
default_n_consumer_clusters = 1000
n_labels = 8
keep_feats += list(map(str, range(n_labels)))
TOTAL_VALUE = 'TOTAL_VALUE'
CLUSTER = 'CLUSTER'
NUMBER_OF_WEEKS = 'NUMBER_OF_WEEKS'
consumer = 'consumer'

# Overwrite with local settings
try:
    from consumer_cluster_streaming.local_settings import *
except ModuleNotFoundError:
    pass


def consumer_description_log(base_name):
    return cdl_root / f'{base_name}.csv'


def get_dfs_from_range(t0, te):
    def imp(bn):
        return import_df(consumer_description_log(bn)).set_index(consumer)

    return [imp(v) for k, v in time_name_ty.items() if t0 <= k < te]


def get_y(t0, te, clusters):
    assert te - t0 == 1, 'This is wrong for te-t0!=1'
    test_data = get_dfs_from_range(t0, te)
    n_pictures = len(clusters.unique())
    y = np.empty(shape=(n_pictures,), dtype=np.float)
    for cluster, dfx in clusters.to_frame().reset_index().groupby(CLUSTER):
        y[cluster] = sum([df.reindex(dfx[consumer], fill_value=0)[TOTAL_VALUE].mean() for df in test_data])

    return y


def get_x(t0, te, clusters):
    train_data = get_dfs_from_range(t0, te)

    n_pictures = len(clusters.unique())
    picture_height = len(train_data)
    picture_width = len(keep_feats)

    x = np.empty(shape=(n_pictures, picture_height, picture_width), dtype=np.float)

    for cluster, dfx in clusters.to_frame().reset_index().groupby(CLUSTER):
        for i, td in enumerate(train_data):
            x[cluster, i, :] = td.reindex(dfx[consumer], fill_value=0)[keep_feats].mean(axis=0)

    return x


class CCSSettings(AbstractSettings):

    @property
    def _default_dict(self):
        return {
            'ta': 2,
            'tb': 1,
            'n_consumer_clusters': default_n_consumer_clusters,
            'incremental_training': True,
        }

    def _assign(self, d):
        self.ta = int(self._pop_or_default(d, 'ta'))
        self.tb = int(self._pop_or_default(d, 'tb'))
        self.n_consumer_clusters = int(self._pop_or_default(d, 'n_consumer_clusters'))
        assert self.n_consumer_clusters >= 0
        self.incremental_training = self._bool(self._pop_or_default(d, 'incremental_training'))


class CCSManager(AbstractLocationsManager):

    @property
    def settings_class(self):
        return CCSSettings

    @property
    def fd_base(self):
        return experiments_root

    @property
    def fd_cluster(self):
        x = f'{self.settings.n_consumer_clusters}.{self.settings.ta}.{self.settings.tb}'
        return self.fd_common / 'clusters' / x

    def fn_clusters(self, t):
        return self.fd_cluster / f'{t}.csv'

    def fn_model(self, t):
        return self.fd / 'models' / f'{t}.model'

    def fn_temp_model(self, t):
        return self.fd / 'temp_models' / f'{t}.model'

    def fn_cluster_predictions(self, t):
        return self.fd / 'cluster_predictions' / f'{t}.csv'

    @property
    def settings(self) -> CCSSettings:
        return self._settings

    @property
    def timestamps(self):
        if self.settings.tb != 1:
            raise NotImplementedError('Some things actually go wrong for tb!=1')
        return [ti for ti in time_name_ty.keys() if ti >= self.settings.ta + self.settings.tb]

    @property
    def _fd_parsed_predictions(self):
        return self.fd / 'parsed_predictions'

    @property
    def fn_all_consumer_predictions(self):
        return self._fd_parsed_predictions / 'consumers.csv'

    @property
    def fn_all_cluster_predictions(self):
        return self._fd_parsed_predictions / 'clusters.csv'

    @property
    def fn_all_turnover_predictions(self):
        return self._fd_parsed_predictions / 'turnover.csv'

    @property
    def fd_gt(self):
        return self.fd_common / 'ground_truths'

    @property
    def fn_all_consumer_ground_truths(self):
        return self.fd_gt / f'consumer_ty{self.settings.tb}.csv'

    @property
    def fn_all_cluster_ground_truths(self):
        return self.fd_gt / f'{self.settings.n_consumer_clusters}.{self.settings.ta}.{self.settings.tb}.csv'

    @property
    def fn_all_turnover_ground_truths(self):
        return self.fd_gt / f'turnover.csv'

    @property
    def fn_results(self):
        return self.fd / 'results.csv'

    @property
    def cluster_ok(self):
        def num_clusters(fn):
            return len(import_sr(fn).unique())

        expected = self.settings.n_consumer_clusters
        real = min(map(num_clusters, list_files(self.fd_cluster)))
        return expected == real

    @property
    def fn_births(self):
        return fn_birth_ty


def get_xy_from_css(ccs_manager: CCSManager, t, polyfit_t=None):
    ta = ccs_manager.settings.ta
    tb = ccs_manager.settings.tb
    clusters = import_sr(ccs_manager.fn_clusters(t if polyfit_t is None else polyfit_t))
    return get_x(t0=t - tb - ta, te=t - tb, clusters=clusters), get_y(t0=t - tb, te=t, clusters=clusters)
