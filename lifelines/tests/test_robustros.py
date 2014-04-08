import unittest

import numpy as np
import numpy.testing as npt

import matplotlib.pyplot as plt

import pandas as pd

from ..estimation import RobustROSFitter


class _base_ROS_Mixin(unittest.TestCase):

    def test_zero_data(self):
        data = self.data.copy()
        data['res'].loc[0] = 0.0
        npt.assert_raises(ValueError, RobustROSFitter, data)

    def test_negative_data(self):
        data = self.data.copy()
        data['res'].loc[0] = -1.0
        npt.assert_raises(ValueError, RobustROSFitter, data)

    def test_data(self):
        self.assertTrue(hasattr(self.ros, 'data'))
        self.assertTrue(isinstance(self.ros.data, pd.DataFrame))

    def test_data_cols(self):
        known_cols = ['modeled', 'res', 'cen']
        npt.assert_array_equal(self.ros.data.columns.tolist(), known_cols)

    def test_debug_attr(self):
        self.assertTrue(hasattr(self.ros, 'debug'))
        self.assertTrue(isinstance(self.ros.data, pd.DataFrame))

    def test_debug_cols(self):
        known_cols = [
            'cen', 'res', 'DLIndex',
            'Rank', 'plot_pos', 'Zprelim',
            'modeled_data', 'modeled'
        ]
        npt.assert_array_equal(self.ros.debug.columns.tolist(), known_cols)

    def test_N_obs(self):
        self.assertTrue(hasattr(self.ros, 'N_obs'))
        npt.assert_equal(self.ros.N_obs, self.data.shape[0])

    def test_N_cen(self):
        self.assertTrue(hasattr(self.ros, 'N_cen'))
        npt.assert_equal(self.ros.N_cen, self.data[self.data.cen].shape[0])

    def test_cohn_attr(self):
        self.assertTrue(hasattr(self.ros, 'cohn'))
        self.assertTrue(isinstance(self.ros.cohn, pd.DataFrame))

    def test_cohn_A(self):
        npt.assert_array_equal(self.known_cohn_A,
                                  self.ros.cohn['A'].values)

    def test_cohn_B(self):
        npt.assert_array_equal(self.known_cohn_B,
                                  self.ros.cohn['B'].values)

    def test_cohn_C(self):
        npt.assert_array_equal(self.known_cohn_C,
                                  self.ros.cohn['C'].values)

    def test_cohn_PE(self):
        npt.assert_array_almost_equal(
            self.known_cohn_PE, self.ros.cohn['PE'], decimal=4
        )

    def test_MR_rosEstimator(self):
        known_fd = np.array([
            3.20, 2.80, 1.42, 1.14, 0.95, 0.81,
            0.68, 0.57, 0.46, 0.35, 1.70, 1.50,
            0.98, 0.76, 0.58, 0.41, 0.90, 0.61,
            0.70, 0.70, 0.60, 0.50, 0.50, 0.50
        ])
        known_fd.sort()

        fdarray = np.array(self.ros.data.modeled)
        fdarray.sort()

        npt.assert_array_almost_equal(known_fd, fdarray, decimal=2)

    def test_dup_index_error(self):
        data = self.data.append(self.data)
        npt.assert_raises(ValueError, RobustROSFitter, data)

    def test_non_dataframe_error(self):
        data = self.data.values
        npt.assert_raises(ValueError, RobustROSFitter, data)

    def test_plot_default(self):
        ax = self.ros.plot()
        self.assertTrue(isinstance(ax, plt.Axes))

    def test_plot_ylogFalse_withAx(self):
        fig, ax = plt.subplots()
        ax = self.ros.plot(ylog=False, ax=ax)


class test_ROS_Helsel_Arsenic(_base_ROS_Mixin):
    '''
    Oahu arsenic data from Nondetects and Data Analysis by Dennis R. Helsel
    (John Wiley, 2005)
    '''
    def setUp(self):
        obs = np.array([3.2, 2.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                        2.0, 2.0, 1.7, 1.5, 1.0, 1.0, 1.0, 1.0,
                        0.9, 0.9, 0.7, 0.7, 0.6, 0.5, 0.5, 0.5])
        cen = np.array([False, False, True, True, True, True, True,
                        True, True, True, False, False, True, True,
                        True, True, False, True, False, False, False,
                        False, False, False])
        self.known_cohn_A = np.array([6.0, 1.0, 2.0, 2.0, np.nan])
        self.known_cohn_B = np.array([0.0, 7.0, 12.0, 22.0, np.nan])
        self.known_cohn_C = np.array([0.0, 1.0, 4.0, 8.0, np.nan])
        self.known_cohn_PE = np.array([1.0, 0.3125, 0.2143, 0.0833, 0.0])
        self.data = pd.DataFrame({'res': obs, 'cen': cen})
        self.ros = RobustROSFitter(self.data, result_col='res',
                                   censorship_col='cen')
        self.ros.fit()



