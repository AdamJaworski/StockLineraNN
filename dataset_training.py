import os
import sys
import time
from train_options import opt
import numpy as np
import torch
import utilities


class DatasetTraining:
    """
    created for csv format:
    time,open,high,low,close,Volume,Color,Plot
    """
    input_list: list
    gt_list: list
    size: int

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
            reference_price, full_csv_data = utilities.normalize_data(full_csv_data)

            # rn data is in % change in compare to previous candle close price
            # so for example {0,02 0,025 -0,01 0,22 50}
            # only rsi is in range 0-100, rest is in %

            open_price, close_price = full_csv_data[opt.CANDLE_INPUT][0], full_csv_data[opt.CANDLE_INPUT][3]
            input_tensor = torch.from_numpy(np.array(full_csv_data[0:opt.CANDLE_INPUT], dtype=np.float32))
            answer_tensor = torch.from_numpy(np.array([open_price, close_price], dtype=np.float32))
            self.input_list.append(input_tensor)
            self.gt_list.append(answer_tensor)

            self.total_size += sys.getsizeof(input_tensor)
            self.total_size += sys.getsizeof(answer_tensor)

            for i in range(opt.CANDLE_INPUT + 1, len(full_csv_data)):
                open_price, close_price = full_csv_data[i][0], full_csv_data[i][3]
                input_tensor = torch.from_numpy(np.array(full_csv_data[(i - opt.CANDLE_INPUT): i], dtype=np.float32))
                answer_tensor = torch.from_numpy(np.array([open_price, close_price], dtype=np.float32))
                self.input_list.append(input_tensor)
                self.gt_list.append(answer_tensor)

                self.total_size += sys.getsizeof(input_tensor)
                self.total_size += sys.getsizeof(answer_tensor)

        self.input_dict = {i: k for i, k in enumerate(self.input_list)}
        self.gt_dict    = {i: k for i, k in enumerate(self.gt_list)}

        self.total_size += sys.getsizeof(self.input_dict)
        self.total_size += sys.getsizeof(self.gt_dict)

        if len(self.input_list) == len(self.gt_list):
            self.size = len(self.gt_list)
            print("Dataset initialized correctly")
        else:
            print("Error while creating dataset")
            sys.exit()

    def get(self, index):
        return self.input_dict[index], self.gt_dict[index]



