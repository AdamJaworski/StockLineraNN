import os
import sys
import numpy as np
import torch


class Dataset:
    """
    created for csv format:
    time,open,high,low,close,Volume,Color,Plot
    """
    input_list: list
    gt_list: list
    size: int

    def __init__(self, data_path):
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

            self.input_list.append(torch.from_numpy(np.array(full_csv_data[0:50])))
            self.gt_list.append(torch.from_numpy(np.array(full_csv_data[50])))

            for i in range(51, len(full_csv_data)):
                self.input_list.append(torch.from_numpy(np.array(full_csv_data[(i - 50): i])))
                self.gt_list.append(torch.from_numpy(np.array(full_csv_data[i])))

        if len(self.input_list) == len(self.gt_list):
            self.size = len(self.gt_list)
            print("Data set initialized correctly")
        else:
            print("Error while creating data set")
            sys.exit()

    def get(self, index):
        return self.input_list[index], self.gt_list[index]