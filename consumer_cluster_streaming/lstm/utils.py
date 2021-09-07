import numpy as np
import pandas as pd


def xy_to_generators(x, y, batch_size, **kwargs):
    from tensorflow.python.keras.utils.data_utils import Sequence

    if isinstance(y, pd.Series):
        y = y.to_numpy()

    if len(x) == 1:
        xt = x
        yt = y
        xv = np.empty(shape=(0, xt.shape[1]))
        yv = np.empty(shape=(0,))
    else:
        from sklearn.model_selection import train_test_split
        xt, xv, yt, yv = train_test_split(x, y, **kwargs)

    class ConsumerSequence(Sequence):

        def __init__(self, _x, _y):
            self._x = _x
            self._y = _y
            self.shape = _x.shape
            if _y.ndim == 1:
                self.n_labels = 1
            else:
                self.n_labels = _y.shape[1]

        def __getitem__(self, index):
            return self._x[index * batch_size: (index + 1) * batch_size], \
                   self._y[index * batch_size: (index + 1) * batch_size]

        def __len__(self):
            return (self.shape[0] + batch_size - 1) // batch_size

    return ConsumerSequence(xt, yt), ConsumerSequence(xv, yv)
