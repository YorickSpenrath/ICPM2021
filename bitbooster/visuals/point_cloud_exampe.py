import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from bitbooster.preprocessing.discretizers import discretize

k = 5
n = 3

coloured_with_label = False

dense_example = False

if coloured_with_label:
    colour_wheel = 'rybcmyk'
else:
    colour_wheel = 'kkkkkkk'


def run():
    data, labels = make_blobs(n_samples=250, n_features=2, centers=k, random_state=684, cluster_std=0.9)

    mms = MinMaxScaler(feature_range=(0, 2 ** n))
    data = mms.fit_transform(data)

    if dense_example:
        data[:240, :] *= 0.5

    plot_data = data * (2 ** n - 1) / (2 ** n)

    def add_point_cloud(axes):
        for lab in range(k):
            axes.plot(plot_data[np.where(labels == lab), 0], plot_data[np.where(labels == lab), 1],
                      '.' + colour_wheel[lab], ms=7)

    def makeup(axes):
        axes.set_xlim(-.5, 2 ** n - .5)
        axes.set_xticks([])
        axes.set_ylim(-.5, 2 ** n - .5)
        axes.set_yticks([])

    # Original frame
    f, ax = plt.subplots()
    add_point_cloud(ax)
    makeup(ax)
    plt.show()

    # Fit stuff
    data = discretize(pd.DataFrame(data), n)

    disc_points = np.array([[[x, y] for x in range(2 ** n)] for y in range(2 ** n)])
    f, ax = plt.subplots()

    def add_disc(axes, **kwargs):
        axes.plot(disc_points[:, :, 0].flatten(), disc_points[:, :, 1].flatten(), 'ko', ms=90 // n, mfc='w', **kwargs)

    add_disc(ax)
    add_point_cloud(ax)

    makeup(ax)
    plt.show()

    alt_points = data.to_numpy(dtype=float)
    alt_points += 2 * (0.1 * np.random.rand(*alt_points.shape) - 0.05)
    f, ax = plt.subplots()
    add_disc(ax)
    for i in range(k):
        ax.plot(alt_points[np.where(labels == i), 0], alt_points[np.where(labels == i), 1], '.' + colour_wheel[i])
    makeup(ax)
    plt.show()


if __name__ == '__main__':
    run()
