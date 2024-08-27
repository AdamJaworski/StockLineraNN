from dataset.dataset_class import Dataset
from settings import global_variables
from dataset.file_deconstructor import File
import torch
import gc
import os
import sys

FALLING = [1, 0, 0]
EVEN    = [0, 1, 0]
RISING  = [0, 0, 1]

class Trend(Dataset):
    def __init__(self, data_path):
        self.total_size = 0
        self.input_list = []
        self.gt_list = []
        print("Creating data set...")
        try:
            data_list = os.listdir(data_path)
        except Exception as e:
            data_list = [data_path]

        for data_file in data_list:
            data_from_file = File(data_path / data_file)

            input_array = data_from_file.get_candles()[0:global_variables.model_settings["candle_input"]]
            price_change = data_from_file.candle_raw[global_variables.model_settings["candle_input"] + global_variables.model_settings["forward_candle_prediction"]][3] / data_from_file.candle_raw[global_variables.model_settings["candle_input"]][3]

            if price_change < 0.99:
                output_array = FALLING
            elif price_change > 1.01:
                output_array = RISING
            else:
                output_array = EVEN


            input_tensor = torch.tensor(input_array, dtype=global_variables.TENSOR_DATA_TYPE, device=global_variables.device)
            answer_tensor = torch.tensor(output_array, dtype=global_variables.TENSOR_DATA_TYPE, device=global_variables.device)
            self.input_list.append(input_tensor)
            self.gt_list.append(answer_tensor)

            self.total_size += sys.getsizeof(input_tensor)
            self.total_size += sys.getsizeof(answer_tensor)

            i = 1
            while i + global_variables.model_settings["candle_input"] + global_variables.model_settings["forward_candle_prediction"] < data_from_file.get_size():
                input_tensor = torch.tensor(data_from_file.get_candles()[i:global_variables.model_settings["candle_input"] + i], dtype=global_variables.TENSOR_DATA_TYPE, device=global_variables.device)
                price_change = data_from_file.candle_raw[global_variables.model_settings["candle_input"] + global_variables.model_settings["forward_candle_prediction"] + i][3]  / data_from_file.candle_raw[global_variables.model_settings["candle_input"] + i][3]

                if price_change < 0.99:
                    output_array = FALLING
                elif price_change > 1.01:
                    output_array = RISING
                else:
                    output_array = EVEN

                answer_tensor = torch.tensor(output_array, dtype=global_variables.TENSOR_DATA_TYPE, device=global_variables.device)
                self.input_list.append(input_tensor)
                self.gt_list.append(answer_tensor)

                self.total_size += sys.getsizeof(input_tensor)
                self.total_size += sys.getsizeof(answer_tensor)

                i += 1

        if len(self.input_list) == len(self.gt_list):
            self.size = len(self.gt_list)
            print("Dataset initialized correctly")
        else:
            print("Error while creating dataset")
            sys.exit()


if __name__ == "__main__":
    dataset = Trend(r'..\Data\DataTraining\GPW_DLY_WIG20, 1D.csv')
    data = File(r'..\Data\DataTraining\GPW_DLY_WIG20, 1D.csv')
    for i in range(25):
        print(dataset.gt_list[i])
        print(data.candle_raw[100 + i][1], data.candle_raw[110 + i][1])
