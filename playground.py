import matplotlib.pyplot as plt
from train_options import opt
import numpy as np


iterations = []
a_loss = []
h_loss = []
l_loss = []
file_path = rf"./models/{opt.MODEL}/loss.txt"

with open(file_path, 'r') as file:
    for line in file:
        if "finished_process" in line:
            parts = line.split()
            # Appending values to their respective lists
            a_loss.append(float(parts[7]))
            h_loss.append(float(parts[9].rstrip(',')))
            l_loss.append(float(parts[11]))


def plot_data(data):
    global a_loss, h_loss, l_loss
    # Re-plotting with the corrected data
    iterations = [*range(0, len(data)*opt.PRINT_RESULTS, opt.PRINT_RESULTS)]

    iterations = iterations[1:]
    data     = data[1:]

    plt.figure(figsize=(14, 8))
    plt.plot(iterations, data, label='loss', marker='o')
    # plt.plot(iterations, h_loss, label='h_loss', marker='x')
    # plt.plot(iterations, l_loss, label='l_loss', marker='^')
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.title('Losses over Iterations')

    z = np.polyfit(iterations, data, 1)
    p = np.poly1d(z)

    plt.plot(iterations, p(iterations))

    plt.legend()
    plt.grid(True)
    plt.show()


plot_data(a_loss)