import itertools
import string
from datetime import datetime
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import XKCD_COLORS

from consumer_cluster_streaming import common as locations
from functions import confidence_interval
from functions.dataframe_operations import import_df, export_df
from functions.general_functions import listified
from functions.progress import ProgressShower
from consumer_cluster_streaming.common import CCSManager

markers = 'oX^sphD8P|<>v'
out = 'pdf', 'svg'
colors = list(XKCD_COLORS)


def k_to_string(__x):
    if np.isfinite(__x):
        return str(int(__x))
    else:
        return r'$|\mathcal{C}|$'


def texify(s):
    return dict(
        ta=r'$\tau$',
        n_consumer_clusters='$k$',
        r2_cluster='Cluster $R^2$',
        rmse_cluster='Cluster $RMSE$',
        turnover_error_cluster=r'$\hat{T}$ relative error',
        absolute_turnover_error_cluster=r'$\hat{T}$ absolute percentage error',
        bottom_absolute_f1_cluster='$F_1$ on top decile',
    ).get(s, s)


def combine_results():
    res = pd.DataFrame()
    for fd in ProgressShower(locations.experiments_root):
        if fd.name == 'common':
            continue
        c = CCSManager(fd.name)
        if not c.fn_results.exists():
            continue
        res = res.append(import_df(c.fn_results).assign(**c.settings.as_dict()).assign(name=fd.name))

    export_df(res, locations.experiments_root / 'common' / 'results.csv')


def get_results():
    df = import_df(locations.experiments_root / 'common' / 'results.csv')

    def fix(c):
        if c.isdigit():
            return int(c)
        return c

    df.columns = map(fix, df.columns)
    return df


def fix_integer_columns(df):
    df.columns = map(int, df.columns)
    return df


def make_title(d):
    s = []

    if 'ta' in d:
        s.append(f'{texify("ta")[:-1]}={d["ta"]}$')

    if 'n_consumer_clusters' in d:
        s.append(f'{texify("n_consumer_clusters")[:-1]}={d["n_consumer_clusters"]}$')

    return ', '.join(s)


class PlotType(Enum):
    time = 'time'
    bar = 'bar'
    line = 'line'
    bar3d = 'bar3d'
    heatmap = 'heatmap'


def save(name: str, metric: str, plot_type: PlotType):
    fd_out = locations.experiments_root / 'common' / 'figures' / name
    fd_out.mkdir(exist_ok=True, parents=True)
    for ext in listified(out, str):
        plt.savefig(fd_out / f'{metric}_{plot_type.value}.{ext}', bbox_inches='tight')


def __get_prepped_df(sorting=None, **kwargs):
    df = get_results()

    for metric in ['turnover_error_cluster', 'turnover_error_consumer']:
        df_abs = df[df.metric == metric].copy()
        df_abs['metric'] = 'absolute_' + metric
        t_values = list(filter(lambda x: isinstance(x, int), df_abs.columns))
        df_abs[t_values] = df_abs[t_values].abs() * 100
        df = df.append(df_abs)

    # filtering data
    for k, v in kwargs.items():
        vi = v
        if not callable(v):
            def v(vx):
                return vx == vi
        df = df[df[k].apply(v)]

    df['n_consumer_clusters'] = df['n_consumer_clusters'].apply(lambda x: x if x != 0 else np.inf)

    # sorting data
    if sorting is not None:
        if isinstance(sorting, dict):
            for k, v in sorting.items():
                if len(df[k].unique()) == 1:
                    continue
                if callable(v):
                    v = sorted(df[k].unique())
                df[f'{k}:index'] = df[k].apply(lambda x: v.index(x))

            sort_cols = list(filter(lambda x: isinstance(x, str) and x.endswith(':index'), df.columns))
            if sort_cols:
                df = df.sort_values(sort_cols).drop(columns=sort_cols)
        elif isinstance(sorting, str):
            df = df.sort_values(sorting)

    return df


def _t_columns(df):
    return sorted(filter(lambda c: isinstance(c, int), df.columns))


def _remove_nan_columns(df):
    skip_mask = (df.metric.str.startswith('r2')) & (df['n_consumer_clusters'] == 1)
    return df.loc[:, ~df[~skip_mask].isna().any(axis=0)]


def do(metric: str, plot_type: PlotType, result_name: str, relative_average: bool = False, group_tx: bool = True,
       sorting=None, ax=None, ls='-',
       **kwargs):
    df = __get_prepped_df(sorting, **kwargs)
    df = df[df.metric == metric]

    # Title and label ==================================================================================================
    constants = dict()
    for i in ['ta', 'incremental_training', 'n_consumer_clusters']:
        if len(set(df[i])) == 1:
            constants[i] = df.iloc[0][i]

    title_add = make_title(constants)

    def make_label(d):
        s = []
        if 'incremental_training' not in constants:
            s += ['I' if d['incremental_training'] else 'R']
        if 'ta' not in constants:
            s += [f'{d["ta"]}']
        if 'n_consumer_clusters' not in constants:
            s += [f'{k_to_string(d["n_consumer_clusters"])}']
        return '.'.join(s)

    # Make plot ========================================================================================================
    if ax is None:
        f, ax = plt.subplots()
        save_figure = True
    else:
        f = None
        save_figure = False

    t_columns = _t_columns(df)

    if plot_type == PlotType.time:
        for i, r in df.reset_index().iterrows():
            x = t_columns
            y = r[t_columns].to_numpy()
            if relative_average:
                y -= r[t_columns].mean()
                # ax.set_yscale('symlog')
                # for _ in [-0.05, 0.05]:
                #     ax.axhline(_, color='k', ls='-', lw=.2, alpha=0.1)
            plot_as_lines(x, y, ax, dot_align='left', marker=markers[i], color=list(XKCD_COLORS.keys())[i],
                          label=make_label(r), ls=ls, connect=True)
        ax.set_title(f'{"Relative to average " if relative_average else ""}{texify(metric)} over time ({title_add})')
        add_events(ax, ano=True, label=False)
        ax.set_xlabel('t')
        ax.legend(ncol=6)
        f.set_size_inches(w=10, h=3.5)

    # Everything below is a mean; so we remove all columns that contain a NaN value; unless that NaN value is only from
    # the r2 score from k=1
    df = _remove_nan_columns(df)
    t_columns = _t_columns(df)

    # This sorts the DataFrame as desired
    line_parameter = 'ta' if group_tx else 'n_consumer_clusters'
    x_parameter = 'n_consumer_clusters' if group_tx else 'ta'
    df = df.sort_values([line_parameter, x_parameter])

    if plot_type == PlotType.line:
        for i, (line_value, line_df) in enumerate(df.groupby(line_parameter)):
            x = line_df[x_parameter]
            y = line_df[t_columns].mean(axis=1)
            ax.loglog(x, y, marker=markers[i], color=list(XKCD_COLORS.keys())[i],
                      label=f'{texify(line_parameter)}={line_value}', ls=ls)
        ax.set_xticks(df[x_parameter].unique())
        ax.set_xticklabels(df[x_parameter].astype(str).unique(), rotation=90)
        ax.set_title(f'time-average {metric} ({title_add})')
        ax.legend(ncol=4)

    if plot_type == PlotType.bar:
        group_number = {k: i for i, k in enumerate(sorted(df[line_parameter].unique()))}
        for i, r in df.reset_index().iterrows():
            c = colors[group_number[r[line_parameter]]]
            x_position = i + group_number[r[line_parameter]]
            if all(r[t_columns].isna()):
                m = 0
                r = 0
            else:
                m, r = confidence_interval.get_mean_and_ci(r[t_columns].values, .95)

            if m <= 0:
                ax.text(x_position, 0, 'x', ha='center', va='center', color=c)
            else:
                ax.bar(x=x_position, height=m, yerr=r, color=c, error_kw=dict(lw=.5, color='k'))

    # Don't report r2 for k=1 in the heatmap / 3d bar
    if metric.startswith('r2'):
        df = df[df['n_consumer_clusters'] > 1]

    data = df.set_index(['ta', 'n_consumer_clusters'])[t_columns].mean(axis=1).sort_index().unstack()

    if plot_type == PlotType.bar3d:
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        _xx, _yy = np.meshgrid(np.arange(len(data.index)), np.arange(len(data.columns)))
        x, y = _xx.ravel(), _yy.ravel()

        ax.bar3d(x, y, np.zeros_like(x), 1, 1, np.maximum(0, data.values.ravel()), shade=True,
                 # color=['k', 'r'] + list('kkkk')
                 )
        ax.set_zlim(0, np.nanmax(data.values))
        ax.set_xticks(np.arange(len(data.index)))
        ax.set_xticklabels(data.index)
        ax.set_yticks(np.arange(len(data.columns)))
        ax.set_yticklabels(map(str, data.columns))

    if plot_type == PlotType.heatmap:
        raise NotImplementedError

    if save_figure:
        save(result_name, metric, plot_type)
        plt.close(f)


def _get_heatmap_parameters(metric, data):
    invert, num_dec, take_abs = False, 2, False

    if metric.startswith('absolute_turnover_error'):
        invert = True
        num_dec = 0
    elif metric.startswith('r2'):
        pass
    elif metric.startswith('rmse'):
        invert = True
        num_dec = 0
    elif metric.startswith('bottom_'):
        pass
    else:
        raise NotImplementedError(metric)

    if take_abs:
        data_for_colours = np.abs(data.values)
    else:
        data_for_colours = np.maximum(0, data.values)

    # Scale
    min_value = np.nanmin(data_for_colours)
    max_value = np.nanmax(data_for_colours)
    non_na = data_for_colours[np.where(~np.isnan(data_for_colours))]

    data_for_colours[np.where(~np.isnan(data_for_colours))] = (non_na - min_value) / (max_value - min_value)

    if invert:
        data_for_colours = 1 - data_for_colours

    skip_zero = (np.isnan(data.to_numpy()) | (np.abs(data.to_numpy()) < 1)).all()

    return num_dec, data_for_colours, skip_zero


def make_heat(ax, data, data_for_colours, num_dec, skip_zero, df_ci=None):
    """

    Parameters
    ----------
    ax
    data
    data_for_colours
    num_dec: int or list of int
    df_ci: None or pd.DataFrame
    skip_zero

    Returns
    -------

    """
    if isinstance(num_dec, int):
        num_dec = [num_dec] * len(data)
    if isinstance(data_for_colours, pd.DataFrame):
        data_for_colours = data_for_colours.to_numpy()

    base = cm.get_cmap('inferno', 1000)

    if df_ci is None:
        aspect = .45
        font_size = 8
    else:
        aspect = .35
        font_size = 6

    # def mapper(s):
    #     if not isinstance(s, int):
    #         return s
    #     if s % 1000 == 0:
    #         return f'{s // 1000}K'
    #     else:
    #         return s

    ax.imshow(data_for_colours, cmap=base)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontdict=dict(size=font_size))
    ax.set_ylabel(texify('ta'))
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(map(k_to_string, data.columns), fontdict=dict(size=font_size))
    ax.set_xlabel(texify('n_consumer_clusters'))

    for x in range(len(data.columns)):
        for y in range(len(data.index)):
            if pd.isna(data.iloc[y, x]):
                continue
            c = 'w' if (data_for_colours[y, x] < 0.5) else 'k'
            t = f'{data.iloc[y, x]:.{num_dec[y]}f}'
            if skip_zero[y]:
                t = t.replace('0.', '.')
            if df_ci is not None:
                t = t + rf'$\pm$'
                t_ci = f'{df_ci.iloc[y, x]:.{num_dec[y]}f}'
                if skip_zero[y]:
                    t_ci = t_ci.replace('0.', '.')
                t = t + t_ci
            ax.text(x, y, t, color=c, ha='center', va='center', fontdict=dict(size=font_size))

    ax.set_aspect(aspect=aspect)


def make_combined_plot(cluster_names, metric='rmse_cluster', **kwargs):
    df = __get_prepped_df(sorting=dict(name=cluster_names), name=lambda x: x in cluster_names,
                          metric=lambda x: x == metric)

    df = df.assign(color=list('grbcymk'[:len(df)]), marker=list('^vDop')[:len(df)])

    t_columns = _t_columns(df)
    f, ax = plt.subplots()
    for i, r in df.reset_index().iterrows():
        label = rf'$t_x={r["ta"]}, k={r["n_consumer_clusters"]}$'
        _kwargs = dict(marker=r['marker'], color=r['color'], ls=':', label=label, alpha=0.6)
        _kwargs.update(kwargs)
        # ax.plot(t_columns, r[t_columns], **kwargs)
        plot_as_lines(ax=ax, x_left=t_columns, y=r[t_columns], **_kwargs, connect=True, dot_align='left')
    ax.set_title(texify(metric) + r" over time")
    ax.set_xlabel('t')
    ax.set_ylabel(texify(metric))
    if metric.startswith('turnover_error_'):
        ax.axhline(y=0, color='k', ls=':', alpha=1 / 3, lw=1 / 3)
    add_events(ax, ano=True, label=False, alpha=1 / 3, ls='-', lw=1 / 3)

    # legend
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.15,
                     box.width, box.height * 0.85])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(cluster_names), prop=dict(size=9))
    f.set_size_inches(w=8, h=4)

    plt.savefig(locations.experiments_root / f'common/figures/time_{metric}.pdf', bbox_inches='tight')
    plt.savefig(locations.experiments_root / f'common/figures/time_{metric}.svg', bbox_inches='tight')
    plt.close(f)


def multi_heat(metrics, name='heat', ci=None, **kwargs):
    """

    Parameters
    ----------
    metrics: iterable of str
        Which metrics to report, in the correct order
    name: str
        Output file base name
    ci: None or float
        If not None, report this CI level

    Other Parameters
    ----------------
    Additional kwargs are passed to filter df

    Returns
    -------

    """
    assert 'metric' not in kwargs
    kwargs['metric'] = lambda x: x in metrics
    dfx = __get_prepped_df(sorting={'metric': metrics}, **kwargs)

    dfx = _remove_nan_columns(dfx)

    def adapt(dfz):
        dfz = dfz.drop(columns='metric').set_index(['ta', 'n_consumer_clusters'])[_t_columns(dfx)]
        if ci is None:
            return dfz.mean(axis=1).sort_index().unstack(), None
        else:
            df_m, df_ci = confidence_interval.get_mean_and_ci(dfz.T, ci_level=ci)
            return df_m.sort_index().unstack(), df_ci.sort_index().unstack()

    dfs = {m: adapt(df) for m, df in dfx.groupby('metric', sort=False)}

    sample_df = next(iter(dfs.values()))[0]
    num_tau_values = len(sample_df)
    num_k_values = len(sample_df.columns)

    data = pd.DataFrame()
    data_ci = pd.DataFrame()
    data_for_colours = pd.DataFrame()
    num_dec = []
    skip_zero = []
    empty_row = pd.DataFrame(index=[''], columns=sample_df.columns)

    for m in metrics:
        # Empty
        for i in range(2):
            data = data.append(empty_row)
            data_for_colours = data_for_colours.append(empty_row)
            data_ci = data_ci.append(empty_row)
            num_dec += [None]
            skip_zero += [None]

        data_m = dfs[m][0]
        data_ci_m = dfs[m][1]
        if data_ci_m is not None:
            data_ci = data_ci.append(data_ci_m)
        data = data.append(data_m)
        foo, bar, sz = _get_heatmap_parameters(m, data_m)
        num_dec += [foo] * len(data_m)
        skip_zero += [sz] * len(data_m)
        data_for_colours = data_for_colours.append(pd.DataFrame(bar, columns=data_m.columns, index=data_m.index))

    assert len(data) == len(data_for_colours)
    assert len(data) == len(num_dec)
    if ci is None:
        data_ci = None
    else:
        assert len(data_ci) == len(data)

    f, ax = plt.subplots()
    make_heat(ax, data, data_for_colours, num_dec, df_ci=data_ci, skip_zero=skip_zero)
    for i, m in enumerate(metrics):
        y_pos = (num_tau_values + 2) * i + 0.5
        x_pos = num_k_values / 2
        ax.text(x=x_pos, y=y_pos, s=f'({string.ascii_uppercase[i]}) {texify(m)}', ha='center', va='center')
        if i != 0:
            ax.axhline((num_tau_values + 2) * i - .5, color='k', lw=.5)
    a = 0.5
    ax.tick_params(left=False, bottom=False)
    f.set_size_inches(w=len(data.columns) * a, h=len(data) / 2 * a)
    plt.savefig(locations.experiments_root / f'common/figures/{name}.pdf', bbox_inches='tight')
    plt.savefig(locations.experiments_root / f'common/figures/{name}.svg', bbox_inches='tight')
    plt.show()


def exp5():
    # Best, Worst
    # v = ['retrain_2.6000', 'retrain_4.3000']
    # do(result_name=f'time_cluster',
    #    plot_type=PlotType.time,
    #    name=lambda x: x in v,
    #    sorting=dict(name=v),
    #    incremental_training=True)
    #
    # # Best, Worst
    # v = ['retrain_9.3000', 'retrain_4.3000']
    # do(result_name=f'time_consumer',
    #    plot_type=PlotType.time,
    #    name=lambda x: x in v,
    #    sorting=dict(name=v)
    #    incremental_training=True)

    bottom_combinations = itertools.product(['cluster', 'consumer'], ['absolute', 'relative'], ['f1'])
    bottom_metrics = [f'bottom_{abs_rel}_{met}_{clu_con}' for clu_con, abs_rel, met in bottom_combinations]

    other_metrics = ['r2_cluster', 'rmse_cluster', 'absolute_turnover_error_consumer',
                     'absolute_turnover_error_cluster', 'r2_consumer', 'rmse_consumer']
    for m in other_metrics + bottom_metrics:
        do(metric=m,
           result_name=f'heat',
           plot_type=PlotType.heatmap,
           incremental_training=True)

    v_cluster = ['retrain_3.8000', 'retrain_4.4000', 'retrain_9.1']
    make_combined_plot(cluster_names=v_cluster, metric='turnover_error_cluster', ls='-')

    v_cluster = ['retrain_2.8000', 'retrain_2.2000', 'retrain_10.2000', 'retrain_10.8000']
    make_combined_plot(cluster_names=v_cluster, metric='rmse_cluster', ls=':')


def add_events(ax, ano=False, label=True, **kwargs):
    dates_colours_labels = [
        (datetime(2017, 9, 5), 'r', 'Programme Start'),
        (datetime(2017, 11, 23), 'b', 'Thanksgiving'),
        (datetime(2017, 12, 25), 'g', 'Christmas'),
    ]

    ret = []

    for i, (d, c, l) in enumerate(dates_colours_labels):
        kwargs.setdefault('ls', ':')
        if label:
            kwargs['label'] = f'Event {i + 1}' if ano else l
        ret.append(ax.axvline((d - datetime(2017, 4, 24)).days / 7, color=c, **kwargs))

    return ret


def plot_as_lines(x_left, y, ax, delta=1, connect=False, dot_align='center', **kwargs):
    x_left = np.array(x_left)
    x_right = x_left + delta
    if dot_align == 'center':
        x_mid = x_left + delta / 2
    elif dot_align == 'left':
        x_mid = x_left
    elif dot_align == 'right':
        x_mid = x_right
    else:
        raise NotImplementedError('dot_align parameter must be in [center, left, right]')

    dot_kwargs = kwargs.copy()
    dot_kwargs['ls'] = ''

    line_kwargs = kwargs.copy()
    line_kwargs['marker'] = None

    if dot_kwargs.get('marker', None) is not None:
        line_kwargs.pop('label', None)
    else:
        dot_kwargs.pop('label', None)

    ax.plot(x_mid, y, **dot_kwargs)
    for xl, xr, yp, in zip(x_left, x_right, y):
        ax.plot([xl, xr], [yp, yp], **line_kwargs)
        line_kwargs.pop('label', None)

    if connect:
        line_kwargs['alpha'] = line_kwargs.get('alpha', 1) / 2
        for xp, yt, yb in zip(x_left[1:], y[:-1], y[1:]):
            ax.plot([xp, xp], [yt, yb], **line_kwargs)


def run():
    # combine_results()
    # kwargs = dict(metrics=['rmse_cluster', 'bottom_absolute_f1_cluster', 'absolute_turnover_error_cluster'],
    #               incremental_training=True,
    #               n_consumer_clusters=lambda k: (k in [1, 250, 500, 750, 1000, 0] or k % 2000 == 0))
    #
    # multi_heat(ta=lambda ta: ta % 2 == 0, ci=.95,
    #            name='heat_even', **kwargs)
    #
    # multi_heat(ci=.95, **kwargs)
    for m in ['rmse_cluster', 'bottom_absolute_f1_cluster', 'turnover_error_cluster']:
        do(metric=m, result_name='line', plot_type=PlotType.time, incremental_training=True,
           ta=10, n_consumer_clusters=lambda k: (k in [1, 500, 1000, 0] or k % 4000 == 0),
           sorting='n_consumer_clusters',
           relative_average=False)


if __name__ == '__main__':
    run()
