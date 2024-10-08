import os
import sys
import numpy as np
import torch
from utlities import utilities
from dataset.dataset_class import Dataset
from settings import global_variables

class OpenClose(Dataset):
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
            csv_data = np.loadtxt(data_path / csv, delimiter=',', dtype=str)
            for index, row in enumerate(csv_data):
                if index == 0:
                    continue
                full_csv_data.append(row[1:5].astype('float32'))

            rsi = utilities.get_rsi(full_csv_data)
            for i, candle in enumerate(full_csv_data):
                if rsi[i] > 100 or rsi[i] < 0:
                    print(csv, rsi, candle)
                full_csv_data[i] = np.append(candle, rsi[i] / 100)

            full_csv_data = np.array(full_csv_data[4:], dtype=np.float32)
            # struct at this point: { open, high, low, close, rsi }
            reference_price, full_csv_data = utilities.normalize_data(full_csv_data)

            # rn data is in % change in compare to previous candle close price
            # so for example {0,02 0,025 -0,01 0,22 50}
            # only rsi is in range 0-100, rest is in %

            open_price, close_price = full_csv_data[global_variables.model_settings["candle_input"]][0], full_csv_data[global_variables.model_settings["candle_input"]][3]

            input_tensor = torch.from_numpy(np.array(full_csv_data[0:global_variables.model_settings["candle_input"]], dtype=np.float32))
            answer_tensor = torch.from_numpy(np.array([open_price, close_price], dtype=np.float32))
            self.input_list.append(input_tensor)
            self.gt_list.append(answer_tensor)

            self.total_size += sys.getsizeof(input_tensor)
            self.total_size += sys.getsizeof(answer_tensor)

            for i in range(global_variables.model_settings["candle_input"] + 1, len(full_csv_data)):
                open_price, close_price = full_csv_data[i][0], full_csv_data[i][3]
                input_tensor = torch.from_numpy(np.array(full_csv_data[(i - global_variables.model_settings["candle_input"]): i], dtype=np.float32))
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

