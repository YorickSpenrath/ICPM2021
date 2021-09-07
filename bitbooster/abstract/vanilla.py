from abc import ABC

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix

from bitbooster.abstract.clusterable import Clusterable


class BaseVanilla(Clusterable, ABC):

    @property
    def feature_colours(self):
        return {k: 'b' for k in self.column_names}

    def series_to_labels(self, labels):
        if isinstance(labels, pd.Series):
            if labels.index.dtype == str:
                labels.reindex(self.names)
            labels = labels.map({l: i for i, l in enumerate(labels.unique())}).to_numpy()
        return labels

    def cluster_and_visualize_features(self, **kwargs):
        medoid_indexes, labels = self.cluster(**kwargs)
        self.visualize_features(labels)

    def visualize_features(self, labels):
        k = len(labels.unique())
        labels = self.series_to_labels(labels)
        if k <= 10:
            colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colours = pd.Series(labels).map({i: c for i, c in enumerate(colours)})
            ax_mat = scatter_matrix(self.dataframe, grid=True, color=colours, diagonal='hist')
            df = self.dataframe.assign(label=labels)
            for i in range(ax_mat.shape[0]):
                ax = ax_mat[i, i]
                assert isinstance(ax, plt.Axes)
                ax.clear()
                for lab, df_lab in df.groupby('label'):
                    ax.hist(x=df_lab[self.column_names[i]], alpha=0.5, color=colours[lab])
            plt.show()

    def visualize_features2(self, labels, show_top, percentile=0):
        nf = self.data.shape[1]
        k = len(labels.unique())
        labels = self.series_to_labels(labels)
        df = self.dataframe.assign(label=labels)

        percentile = float(percentile)
        assert 0 <= percentile < 0.5
        min_percentile = percentile
        max_percentile = 1 - percentile

        label_names, label_counts = np.unique(labels, return_counts=True)
        foo = [(x, y) for x, y in zip(label_names, label_counts)]
        foo = [t[0] for t in sorted(foo, key=lambda t: t[1], reverse=True)]
        df = df[df.label.isin(foo)]
        rename_dict = {y: x for x, y in enumerate(foo)}
        df.label = df.label.map(rename_dict)
        df = df.sort_values('label')

        if show_top is None:
            show_top = k
        elif isinstance(show_top, float):
            assert 0 < show_top <= 1, 'If show_top is a float, it must be in (0,1]'
            x = df.label.value_counts().cumsum() / len(df)
            show_top = min(x[x >= show_top].index)
            df = df[df.label < show_top]
        elif isinstance(show_top, int):
            assert 0 < show_top <= k, 'If show_top is an int, it must be in 1 ... k'
            df = df[df.label < show_top]
        else:
            raise ValueError('show_top must be int, float or None')

        f, ax_mat = plt.subplots(nrows=show_top, ncols=nf)

        for lab, dfl in df.groupby('label'):
            for ax_feature_index, feature_name in enumerate(self.column_names):
                xmin = np.quantile(df[feature_name], min_percentile)
                xmax = np.quantile(df[feature_name], max_percentile)

                ax = ax_mat[lab, ax_feature_index]
                assert isinstance(ax, plt.Axes)
                parts = ax.violinplot(dataset=dfl[feature_name], vert=False, showmeans=True)

                for pc in parts['bodies']:
                    pc.set_facecolor(self.feature_colours[feature_name])
                    pc.set_alpha(1)

                for x in ['cmins', 'cmaxes', 'cbars', 'cmeans']:
                    parts[x].set_edgecolor('k')

                # X axis
                ax.set_xlim(xmin, xmax)
                if lab == show_top - 1:
                    ax.set_xticks([xmin, xmax])
                    ax.set_xticklabels(['LOW', 'HIGH'])
                else:
                    ax.set_xticks([])

                # Y axis
                ax.set_yticks([])
                ax.set_ylim()
                if ax_feature_index == 0:
                    ax.set_ylabel(rf'$C_{lab}$' + f'[{len(dfl)}]', fontsize=15)

                # Title
                if lab == 0:
                    ax.set_title(feature_name, fontsize=20)

        f.set_size_inches(w=4 * nf, h=2 * show_top)

        plt.show()


def cluster_and_visualize_features2(self, show_top=None, **kwargs):
    medoid_indices, labels = self.cluster(**kwargs)
    return self.visualize_features2(labels, show_top)
