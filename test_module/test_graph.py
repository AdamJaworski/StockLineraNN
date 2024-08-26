import os
from model.model import Model
import torch
import matplotlib.gridspec as gridspec
from utlities import utilities
from dataset.dataset_test import DatasetTest
from model.model import Model
import numpy as np
import matplotlib.pyplot as plt


def test_graph(model, data_file):
    data_dir = './Data/DataTraining/'
    dataset  = DatasetTest(data_dir + data_file)

    base_price, gt_list = dataset.get()

    nn_output    = np.array([[candle[0], candle[3]] for candle in gt_list[:opt.CANDLE_INPUT]])
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
        print("One of the arrays has zero variance. Correlation cannot be calculated.")
    else:
        # Ensure both arrays have the same length
        min_length = min(len(price_base_gt), len(price_base_nn))
        price_base_gt = price_base_gt[:min_length]
        price_base_nn = price_base_nn[:min_length]

        # Calculate correlation
        correlation_matrix = np.corrcoef(price_base_gt, price_base_nn)
        print(f"Correlation coefficient: {correlation_matrix[0, 1]}")

    compare_list = compare_list.reshape(compare_list.size, 1)
    price_base_gt = price_base_gt.reshape(price_base_gt.size, 1)
    nn_output = nn_output.reshape(nn_output.size, 1)
    price_base_nn = price_base_nn.reshape(price_base_nn.size, 1)
    iterations = [*range(compare_list.size)]
    to_rsi_list = [price_base_gt[i*2 + 1] for i in range(int(len(price_base_gt) / 2))]
    to_rsi_list = np.array(to_rsi_list).flatten()
    rsi = utilities.get_rsi(to_rsi_list, False)

    plt.figure(figsize=(14, 8))
    plt.plot(iterations, compare_list, label='gt', marker='o')
    plt.plot(iterations, nn_output, label='output', marker='x')
    plt.xlabel('Iterations')
    plt.ylabel('Price')
    plt.title('% Change over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()

    gs = gridspec.GridSpec(4, 1)

    ax1 = plt.subplot(gs[:3, 0])
    ax1.plot(iterations, price_base_gt, label='gt', color='blue')
    ax1.plot(iterations, price_base_nn, label='output', color='orange')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Price')
    ax1.set_title('Price over Iterations')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(gs[3, 0])
    ax2.plot([*range(len(rsi))], rsi, label='RSI', color='purple')
    ax2.set_xlabel('Iterations')
    ax2.set_ylabel('RSI')
    ax2.set_title('RSI')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
