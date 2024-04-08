import os
import sys
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
                full_csv_data[i] = np.append(candle, rsi[i])

            # struct at this point: { open, high, low, close, vol, rsi }

            open_price, close_price = full_csv_data[150][0], full_csv_data[150][3]
            input_tensor = torch.from_numpy(np.array(full_csv_data[0:150], dtype=np.float32))
            answer_tensor = torch.from_numpy(np.array([open_price, close_price], dtype=np.float32))
            self.input_list.append(input_tensor)
            self.gt_list.append(answer_tensor)

            self.total_size += sys.getsizeof(input_tensor)
            self.total_size += sys.getsizeof(answer_tensor)

            for i in range(151, len(full_csv_data)):
                open_price, close_price = full_csv_data[i][0], full_csv_data[i][3]
                input_tensor = torch.from_numpy(np.array(full_csv_data[(i - 150): i], dtype=np.float32))
                answer_tensor = torch.from_numpy(np.array([open_price, close_price], dtype=np.float32))
                self.input_list.append(input_tensor)
                self.gt_list.append(answer_tensor)

                self.total_size += sys.getsizeof(input_tensor)
                self.total_size += sys.getsizeof(answer_tensor)

        if len(self.input_list) == len(self.gt_list):
            self.size = len(self.gt_list)
            print("Dataset initialized correctly")
        else:
            print("Error while creating dataset")
            sys.exit()

    def get(self, index):
        return self.input_list[index], self.gt_list[index]

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