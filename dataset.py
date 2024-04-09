import os
import sys
import time
from train_options import opt
import numpy as np
import torch
import utilities


class Dataset:
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
                full_csv_data.append(row[1:6].astype('float32'))

            rsi = self.get_rsi(full_csv_data)
            for i, candle in enumerate(full_csv_data):
                candle = candle[:len(candle) - 1]
                full_csv_data[i] = np.append(candle, rsi[i])

            full_csv_data = np.array(full_csv_data[4:], dtype=np.float32)

            # struct at this point: { open, high, low, close, rsi }
            reference_price, full_csv_data = self.normalize_data(full_csv_data)

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

    def get_rsi(self, data) -> list:
        close_prices = np.array([candle[4] for candle in data])
        rsi_values = []
        for i in range(len(close_prices)):
            rsi_values.append(utilities.calculate_rsi(close_prices[:i + 1]))

        i = 0
        while rsi_values[i] == -1:
            i += 1

        y = 0
        while rsi_values[y] == -1:
            rsi_values[y] = rsi_values[i]
            y += 1

        return rsi_values

    def normalize_data(self, data):
        for i in reversed(range(len(data))):
            if i == 0:
                continue
            reference_candle = data[i - 1]
            candle_to_normalize = data[i]
            new_candle = (candle_to_normalize - reference_candle[3]) / reference_candle[3]
            new_candle = new_candle[:len(new_candle) - 1]
            new_candle = np.append(new_candle, candle_to_normalize[len(candle_to_normalize) - 1])
            data[i] = new_candle.astype(np.float16)

        reference_price = data[0][3]
        data = data[1:]
        return reference_price, data