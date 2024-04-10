import os
import sys
import time
from train_options import opt
import numpy as np
import torch
import utilities


class DatasetTest:
    """
    created for csv format:
    time,open,high,low,close,Volume,Color,Plot
    """

    def __init__(self, data_path):
        self.total_size = 0
        self.input_list = []
        self.gt_list    = []
        print("Creating data set...")
        csv_list = os.listdir(data_path)

        for csv in csv_list:
            full_csv_data = []
            csv_data = np.loadtxt(data_path + csv, delimiter=',', dtype=str)
            for index, row in enumerate(csv_data):
                if index == 0:
                    continue
                full_csv_data.append(row[1:5].astype('float32'))

            rsi = utilities.get_rsi(full_csv_data)
            for i, candle in enumerate(full_csv_data):
                full_csv_data[i] = np.append(candle, rsi[i])

            full_csv_data = np.array(full_csv_data[4:], dtype=np.float32)
            # struct at this point: { open, high, low, close, rsi }
            self.reference_price, self.gt_list = utilities.normalize_data(full_csv_data)

            # rn data is in % change in compare to previous candle close price
            # so for example {0,02 0,025 -0,01 0,22 50}
            # only rsi is in range 0-100, rest is in %

            print("Dataset initialized correctly")

    def get(self):
        return self.reference_price, self.gt_list



