from unittest.case import TestCase

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from bitbooster.preprocessing.binarizer import binarize
from bitbooster.preprocessing.discretizers import discretize


class TestEqualCutDiscretizer(TestCase):

    @staticmethod
    def _create_equal_cut(n_features, n_datapoints_per_value, n):
        np.random.seed(n_features + n_datapoints_per_value ** n)
        values = 2 ** n

        column_names = sorted([f'f{i:02}' for i in range(n_features)])
        data = np.random.rand(n_datapoints_per_value * values, n_features)
        actual_result = np.ones(shape=data.shape)
        data[0, :] = 0
        for i in range(values):
            data[i * n_datapoints_per_value:(i + 1) * n_datapoints_per_value, :] += i
            actual_result[i * n_datapoints_per_value:(i + 1) * n_datapoints_per_value, :] *= i
        data[-1, :] = values

        df = pd.DataFrame(columns=column_names, data=data)

        for i, a in enumerate(df.columns):
            df[a] = df[a] * (i + 1)

        return df, actual_result

    def test_discretize_equal_cut(self):
        n = 5
        df, actual_result = self._create_equal_cut(100, 10, n)
        m = discretize(df, n)
        assert_array_equal(m.to_numpy(), actual_result)

    # def test_ecd_fit(self):
    #     arr = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    #
    #     ecd = EqualCutDiscretizer(2)
    #     ecd.fit_from_data(arr)
    #     assert_array_equal(ecd.ranges, np.array([1, 1.5, 2]))


#
# class TestJenksDiscretizer(TestCase):
#
#     def test_discretize_jenks_correct_ranges(self):
#         arr = np.array([1, 1, 1, 1, 2, 2, 2, 2])
#         j = JenksDiscretizer(n_bins=2)
#         j.fit_from_data(arr)
#         assert_array_equal(j.ranges, np.array([1, 1, 2]))
#
#     def test_discretize_jenks_correct_transform_series(self):
#         arr = np.array([1, 1, 1, 1, 2, 2, 2, 2])
#         j = JenksDiscretizer(n_bins=2)
#         j.fit_from_data(arr)
#         assert_array_equal(np.array([0, 0, 0, 0, 1, 1, 1, 1]), j.transform(pd.Series(arr)).to_numpy())
#
#     def test_discretize_jenks_correct_transform_numpy(self):
#         arr = np.array([1, 1, 1, 1, 2, 2, 2, 2])
#         j = JenksDiscretizer(n_bins=2)
#         j.fit_from_data(arr)
#         assert_array_equal(np.array([0, 0, 0, 0, 1, 1, 1, 1]), j.transform(arr))
#
#     def test_discretize_jenks_correct_ranges_from_frequencies(self):
#         sr = pd.Series(index=[1, 2], data=[4, 4])
#         j = JenksDiscretizer(n_bins=2)
#         j.fit_from_frequencies(sr)
#         assert_array_equal(np.array([0, 0, 0, 0, 1, 1, 1, 1]), j.transform(np.array([1, 1, 1, 1, 2, 2, 2, 2])))
#
#     def test_discretize_jenks_correct_ranges_from_frequencies2(self):
#         sr = pd.Series(index=range(1, 11), data=range(1, 11))
#         j = JenksDiscretizer(n_bins=2)
#         j.fit_from_frequencies(sr)
#         assert_array_equal(np.array([1, 6, 10]), j.ranges)
#
#
# class TestHeadTailsBreakBinsDiscretizer(TestCase):
#
#     def test_htb_fit(self):
#         arr = np.array([1, 1, 1, 1, 2, 2, 2, 2])
#         htb_bins = HeadTailsBreakBinsDiscretizer(n_bins=2)
#         htb_bins.fit_from_data(arr)
#         assert_array_equal([1, 1.5, 2], htb_bins.ranges)
#
#     def test_htb_transform(self):
#         arr = np.array([1, 1, 1, 1, 2, 2, 2, 2])
#         htb_bins = HeadTailsBreakBinsDiscretizer(n_bins=2)
#         htb_bins.fit_from_data(arr)
#         assert_array_equal(htb_bins.transform(arr), [0, 0, 0, 0, 1, 1, 1, 1])
#
#     def test_htb_fit2(self):
#         sr = pd.Series(range(1, 11), range(1, 11))
#         htb_bins = HeadTailsBreakBinsDiscretizer(n_bins=3)
#         htb_bins.fit_from_frequencies(sr)
#         assert_array_equal([1, 7, 245 / 27, 10], htb_bins.ranges)
#
#
# class TestRatioEqualCutDiscretizer(TestCase):
#
#     def test_recd_fit(self):
#         arr = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3])
#         recd = RatioEqualCutDiscretizer(n_bins=2, f=0.9)
#         recd.fit_from_data(arr)
#         assert_array_equal([1, 1.5, 2], recd.ranges)
#
#
# class TestQCutDiscretizer(TestCase):
#
#     def test_qcut_fit(self):
#         sr = pd.Series(np.arange(1, 11), np.arange(1, 11))
#         qcd = QCutDiscretizer(n_bins=4)
#         qcd.fit_from_frequencies(sr)
#         assert_array_equal([1, 5, 7, 8, 10], qcd.ranges)
#

class TestBinarize(TestCase):

    @staticmethod
    def _generic(arr_in, arr_test, n):
        in_df = pd.DataFrame(data=np.array(arr_in, dtype=int))
        out_df = pd.DataFrame(data=np.array(arr_test, dtype=np.uint64))
        res_df = binarize(in_df, n)
        assert_array_equal(res_df.to_numpy(), out_df.to_numpy())

    def test_binarize1(self):
        self._generic(arr_in=[[1, 2, 3], [0, 1, 2]],
                      arr_test=[[3, 5], [1, 2]],
                      n=2)

    def test_binarize2(self):
        self._generic(arr_in=[[2, 1, 0, 9, 1, 9, 9, 3], [6, 8, 4, 4, 2, 1, 4, 5]],
                      arr_test=[[22, 0, 129, 95], [64, 179, 136, 5]],
                      n=4)
