import os

from nose.tools import *
import numpy as np
import numpy.testing as nptest

import pandas as pd

from ..estimation import RobustROS


class _base_ROS_Mixin:

    @raises(ValueError)
    def test_zero_data(self):
        data = self.data.copy()
        data['res'].loc[0] = 0.0
        ros = RobustROS(data)

    @raises(ValueError)
    def test_negative_data(self):
        data = self.data.copy()
        data['res'].loc[0] = -1.0
        ros = RobustROS(data)

    def test_data(self):
        assert_true(hasattr(self.ros, 'data'))
        assert_is_instance(self.ros.data, pd.DataFrame)

    def test_data_cols(self):
        known_cols = ['modeled', 'res', 'cen']
        assert_list_equal(self.ros.data.columns.tolist(), known_cols)

    def test_debug_attr(self):
        assert_true(hasattr(self.ros, 'debug'))
        assert_is_instance(self.ros.data, pd.DataFrame)

    def test_debug_cols(self):
        known_cols = [
            'cen', 'res', 'DLIndex',
            'Rank', 'plot_pos', 'Zprelim',
            'modeled_data', 'modeled'
        ]
        assert_list_equal(self.ros.debug.columns.tolist(), known_cols)

    def test_N_obs(self):
        assert_true(hasattr(self.ros, 'N_obs'))
        assert_equal(self.ros.N_obs, self.data.shape[0])

    def test_N_cen(self):
        assert_true(hasattr(self.ros, 'N_cen'))
        assert_equal(self.ros.N_cen, self.data[self.data.cen].shape[0])

    def test_cohn_attr(self):
        assert_true(hasattr(self.ros, 'cohn'))
        assert_equal(type(self.ros.cohn), pd.DataFrame)

    def test_cohn_A(self):
        nptest.assert_array_equal(self.known_cohn_A,
                                  self.ros.cohn['A'].values)

    def test_cohn_B(self):
        nptest.assert_array_equal(self.known_cohn_B,
                                  self.ros.cohn['B'].values)

    def test_cohn_C(self):
        nptest.assert_array_equal(self.known_cohn_C,
                                  self.ros.cohn['C'].values)

    def test_cohn_PE(self):
        nptest.assert_array_almost_equal(
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

        nptest.assert_array_almost_equal(known_fd, fdarray, decimal=2)

    @raises(ValueError)
    def test_dup_index_error(self):
        data = self.data.append(self.data)
        ros = RobustROS(data)

    @raises(ValueError)
    def test_non_dataframe_error(self):
        data = self.data.values
        ros = RobustROS(data)


class test_ROS_Helsel_Arsenic(_base_ROS_Mixin):
    '''
    Oahu arsenic data from Nondetects and Data Analysis by Dennis R. Helsel
    (John Wiley, 2005)
    '''
    def setup(self):
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
        self.ros = RobustROS(self.data, rescol='res', cencol='cen')
        self.ros.fit()



