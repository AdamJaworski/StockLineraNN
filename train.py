import torch
import torch.nn as nn
import os
import numpy as np
from model import Model
from custom_loss import CustomLoss
from train_options import opt
from PathManager import PathManager
import pathlib
from datetime import datetime

data_csv = r'./Data/'

"""
created for csv format:
time,open,high,low,close,Volume,Color,Plot
"""


def train_model(model: Model, loss_fn):
    csv_list = os.listdir(data_csv)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR)
    finished_process  = 0
    finished_process_ = 0
    epoch             = opt.STARTING_EPOCH
    running_loss      = 0.0
    highest_loss      = 0
    lowest_loss       = 10000

    while True:
        print(f"Starting epoch: {epoch}")
        np.random.shuffle(csv_list)

        if epoch > 0:
            print(f"Changing LR from: {opt.LR} to {opt.LR / (opt.LR_DROPOFF * epoch)}")
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR / (opt.LR_DROPOFF * epoch))

        for csv in csv_list:
            full_data_list = []
            csv_data = np.loadtxt(data_csv + csv, delimiter=',', dtype=str)
            print(csv)
            for index, row in enumerate(csv_data):
                if index == 0:
                    continue
                full_data_list.append(row[1:6].astype('float32'))
            full_data_list = np.array(full_data_list)

            # Iterating threw csv
            input_array = full_data_list[0:50]
            for i in range(50, len(full_data_list)):
                input_tensor = torch.from_numpy(input_array).unsqueeze(0)
                correct_tensor = torch.from_numpy(full_data_list[i])

                optimizer.zero_grad()

                output_tensor = model(input_tensor)
                output_tensor = output_tensor.squeeze(0)
                loss = loss_fn(output_tensor, correct_tensor)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                finished_process += 1
                finished_process_ += 1
                input_array = full_data_list[(i - 49): (i + 1)]

                if loss.item() < lowest_loss:
                    lowest_loss = loss.item()

                if loss.item() > highest_loss:
                    highest_loss = loss.item()

                if loss.item() < 5:
                    print(f"Loss: {loss.item():.2f}, Output: {output_tensor.detach().numpy()},   GT: {correct_tensor.numpy()}")

                if finished_process % opt.PRINT_RESULTS == 0:
                    print_state(epoch, finished_process, running_loss, finished_process_, highest_loss, lowest_loss)
                    finished_process_, running_loss, highest_loss, lowest_loss = 0, 0, 0, 10000

                if finished_process % opt.SAVE_MODEL_AFTER == 0:
                    save_model(model, str(finished_process))

        epoch += 1


def print_state(epoch, finished_process, running_loss, finished_process_, highest_loss, lowest_loss):
    loss_file = open(model_path_manager.loss_file, 'a+')
    summary_message = f"(epoch: {epoch}, finished_process: {finished_process}) a_loss: {running_loss/finished_process_:.2f} " \
                      f"h_loss: {highest_loss:.2f}, l_loss: {lowest_loss:.2f}"
    print(summary_message)
    loss_file.write(summary_message + '\n')
    loss_file.close()


def save_model(model_instance, name_of_save: str):
    # Current save
    if pathlib.Path.exists(model_path_manager.root_path / (name_of_save + '.pth')):
        pathlib.Path.unlink(model_path_manager.root_path / (name_of_save + '.pth'))
    torch.save(model_instance.state_dict(), model_path_manager.root_path / (name_of_save + '.pth'))

    # Latest save
    if pathlib.Path.exists(model_path_manager.root_path / 'latest.pth'):
        pathlib.Path.unlink(model_path_manager.root_path / 'latest.pth')
    torch.save(model_instance.state_dict(), model_path_manager.root_path / 'latest.pth')


if __name__ == "__main__":
    model = Model()
    model_path_manager = PathManager(opt.MODEL)

    if opt.CONTINUE_LEARNING:
        model.load_state_dict(torch.load(model_path_manager.root_path / 'latest.pth'))

    loss_file = open(model_path_manager.loss_file, 'a+')
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    start_message = '='*55 + f'{dt_string}' + '='*55
    loss_file.write(start_message + '\n')
    loss_file.close()

    print(f"Starting training with rate: {opt.LR}")
    train_model(model, CustomLoss())