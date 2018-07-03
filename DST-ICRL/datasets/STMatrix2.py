from __future__ import print_function
import os
import pandas as pd
import numpy as np

from . import load_stdata
from ..config import Config
from ..utils import string2timestamp


class STMatrix(object):
    """docstring for STMatrix"""

    def __init__(self, data, timestamps, T=48, CheckComplete=True):
        super(STMatrix, self).__init__()
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.pd_timestamps = string2timestamp(timestamps, T=self.T)
        if CheckComplete:
            self.check_complete()
        # index
        self.make_index()

    def make_index(self):
        self.get_index = dict()
        for i, ts in enumerate(self.pd_timestamps):
            self.get_index[ts] = i

    def check_complete(self):
        missing_timestamps = []
        offset = pd.DateOffset(minutes=24 * 60 // self.T)
        pd_timestamps = self.pd_timestamps
        i = 1
        while i < len(pd_timestamps):
            if pd_timestamps[i - 1] + offset != pd_timestamps[i]:
                missing_timestamps.append("(%s -- %s)" % (pd_timestamps[i - 1], pd_timestamps[i]))
            i += 1
        for v in missing_timestamps:
            print(v)
        assert len(missing_timestamps) == 0

    def get_matrix(self, timestamp):
        return self.data[self.get_index[timestamp]]

    def save(self, fname):
        pass

    def check_it(self, depends):
        for d in depends:
            if d not in self.get_index.keys():
                return False
        return True

    def create_dataset(self, len_closeness=3, len_period=3, PeriodInterval=1, len_trend=3, TrendInterval=7, len_y=3):
        """current version
        """
        # offset_week = pd.DateOffset(days=7)
        offset_frame = pd.DateOffset(minutes=24 * 60 // self.T)
        XC = []
        XP = []
        XT = []
        XCY = []
        Y = []
        timestamps_Y = []
        depends = [[-j for j in range(1, len_closeness + 1)[::-1]],
                   [-PeriodInterval * self.T * j for j in range(1, len_period + 1)[::-1]],
                   [-TrendInterval * self.T * j for j in range(1, len_trend + 1)[::-1]],
                   # [-TrendInterval * self.T + j for j in range(len_y)],
                   [j for j in range(len_y)],
                   range(len_y)]

        i = max(self.T * TrendInterval * len_trend, self.T * PeriodInterval * len_period, len_closeness)

        while i < (len(self.pd_timestamps) - (len_y - 1)):
            Flag = True
            for depend in depends:
                if Flag is False:
                    break
                    # Flag = self.check_it([self.pd_timestamps[i] + j * offset_frame for j in depend])

            if Flag is False:
                i += 1
                continue
            x_c = [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in depends[0]]

            x_p = [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in depends[1]]
            x_t = [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in depends[2]]

            def list_diff(a,b):
                return sum(np.sum(np.abs(a[i]-b[i])) for i in range(len(a)))
            minn = 1<<31
            mincnm=0
            now=[self.get_matrix(self.pd_timestamps[i] - j * offset_frame) for j in range(1, 1 + len_closeness)]

            cnm=-1
            while cnm * TrendInterval * self.T-len_closeness+i>=0:
                a = [cnm * TrendInterval * self.T - j for j in range(1, 1 + len_closeness)]

                summ=list_diff(now,[self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in a])

                if minn > summ:
                    minn=summ
                    mincnm=cnm
                cnm = cnm - 1

            cnm=1
            while cnm * TrendInterval * self.T+len_y-1+i<len(self.pd_timestamps):
                a = [cnm * TrendInterval * self.T - j for j in range(1, 1 + len_closeness)]
                summ = list_diff(now, [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in a])
                if minn > summ:
                    minn=summ
                    mincnm=cnm
                cnm = cnm + 1
            print(mincnm)
            a=[mincnm * TrendInterval * self.T + j for j in range(len_y)]
            x_c_y = [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in a]

            # x_c_y = [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in depends[3]]
            y = [self.get_matrix(self.pd_timestamps[i] + j * offset_frame) for j in depends[-1]]
            if len_closeness > 0:
                XC.append((x_c))
            if len_period > 0:
                XP.append((x_p))
            if len_trend > 0:
                XT.append((x_t))
            if len_y > 0:
                XCY.append((x_c_y))
            Y.append(y)
            timestamps_Y.append(self.timestamps[i])
            i += 1
        XC = np.asarray(XC)
        XP = np.asarray(XP)
        XT = np.asarray(XT)
        XCY = np.asarray(XCY)
        Y = np.asarray(Y)
        print("STMatrix  XC shape: ", XC.shape, "XP shape: ", XP.shape, "XT shape: ", XT.shape, "XCY shape: ",
              XCY.shape,
              "Y shape:", Y.shape)
        return XC, XP, XT, XCY, Y, timestamps_Y


if __name__ == '__main__':
    pass
