import matplotlib.pyplot as plt
import pandas as pd

from functions.dataframe_operations import import_sr, import_df
from consumer_cluster_streaming.s99_combined_results import add_events, plot_as_lines
from consumer_cluster_streaming import common as locations
from consumer_cluster_streaming.common import time_name_ty, fn_birth_ty


def print_nr_consumers():
    print(f'{len(import_sr(fn_birth_ty))} unique consumers')


def print_mean_std_visits():
    res = pd.Series(dtype=int)
    for k, v in time_name_ty.items():
        sr = import_df(locations.consumer_description_log(v))['count']
        res = res.add(sr, fill_value=0)

    print(f'Number of visits per consumer: {res.mean():.1f}+-{res.std():.1f}')
    print(f'\tmax={res.max()}, min={res.min()}')


def show_turnover():
    sr = import_sr(locations.experiments_root / 'common' / 'ground_truths' / 'turnover.csv')
    f, ax = plt.subplots()
    plot_as_lines(sr.index, sr.to_numpy(), ax, color='k', ls='-', marker=None, label='Turnover',
                  connect=True)
    ax.set_title('Turnover per week')
    ax.set_xlabel('Week')
    ax.set_ylabel('Turnover')
    ax.set_yticks([])
    add_events(ax, label=True, ano=True, lw=.25)
    ax.legend(ncol=2)
    f.set_size_inches(w=8, h=2)
    plt.savefig(locations.experiments_root / 'common' / 'figures' / 'gt_turnover.pdf', bbox_inches='tight')
    plt.savefig(locations.experiments_root / 'common' / 'figures' / 'gt_turnover.svg', bbox_inches='tight')
    plt.close(f)


def do():
    # print_nr_consumers()
    # print_mean_std_visits()
    show_turnover()


if __name__ == '__main__':
    do()
