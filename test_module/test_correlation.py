import os
from dataset.dataset_test import DatasetTest
import torch
import numpy as np
from model.model import Model

def test_correlation(model):
    data_dir = './Data/DataTraining/'
    list_of_data = os.listdir(data_dir)
    total_correlation = 0

    for data in list_of_data:
        dataset = DatasetTest(data_dir + data)
        base_price, gt_list = dataset.get()

        nn_output = np.array([[candle[0], candle[3]] for candle in gt_list[:opt.CANDLE_INPUT]])
        compare_list = np.array([[candle[0], candle[3]] for candle in gt_list])

        with torch.no_grad():
            i = 1
            while nn_output.shape != compare_list.shape:
                input_tensor = torch.from_numpy(gt_list[i:opt.CANDLE_INPUT + i]).unsqueeze(0)
                output = model(input_tensor)
                nn_output = np.vstack((nn_output, output.numpy().reshape(1, 2)))
                i += 1

        price_base_gt = [base_price + compare_list[0] * base_price]
        price_base_nn = [base_price + nn_output[0] * base_price]

        for i in range(1, compare_list.shape[0]):
            price_base_gt.append(price_base_gt[i - 1][1] + compare_list[i] * price_base_gt[i - 1][1])
            price_base_nn.append(price_base_nn[i - 1][1] + nn_output[i] * price_base_nn[i - 1][1])

        price_base_nn = np.array(price_base_nn)
        price_base_gt = np.array(price_base_gt)

        # Remove NaN or Inf values
        price_base_gt = price_base_gt[~np.isnan(price_base_gt) & ~np.isinf(price_base_gt)]
        price_base_nn = price_base_nn[~np.isnan(price_base_nn) & ~np.isinf(price_base_nn)]

        # Ensure both arrays have variance
        if np.var(price_base_gt) == 0 or np.var(price_base_nn) == 0:
            print(f"{data}: One of the arrays has zero variance. Correlation cannot be calculated.")
        else:
            # Ensure both arrays have the same length
            min_length = min(len(price_base_gt), len(price_base_nn))
            price_base_gt = price_base_gt[:min_length]
            price_base_nn = price_base_nn[:min_length]

            # Calculate correlation
            correlation_matrix = np.corrcoef(price_base_gt, price_base_nn)
            print(f"Correlation coefficient for {data}: {correlation_matrix[0, 1]}")

            total_correlation += 0 if correlation_matrix[0, 1] < 0 else correlation_matrix[0, 1]

    print(f"Total score: {total_correlation}")