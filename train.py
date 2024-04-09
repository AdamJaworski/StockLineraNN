import sys
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import Dataset
from model import Model
from custom_loss import CustomLoss
from train_options import opt
from PathManager import PathManager
import pathlib
from datetime import datetime
import random

data_csv = r'./Data/'


def train_model(model: Model, loss_fn):
    optimizer         = torch.optim.Adam(model.parameters(), lr=opt.LR)
    scheduler         = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    dataset           = Dataset(data_csv)
    finished_process  = 0
    finished_process_ = 0
    epoch             = opt.STARTING_EPOCH
    running_loss      = 0.0
    epoch_loss        = 0.0
    highest_loss      = 0
    lowest_loss       = 10000
    interation        = [*range(dataset.size)]
    print(f"Size of dataset: {dataset.size}/{dataset.total_size / 1024:.2f}KB")
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {num_trainable_params} parameters")

    # print(dataset.get(interation[5]))
    while True:
        print(f"Starting epoch: {epoch}")
        random.shuffle(interation)

        for i in interation:
            input_tensor, correct_tensor = dataset.get(i)
            optimizer.zero_grad()

            input_tensor = input_tensor.unsqueeze(0)
            output_tensor = model(input_tensor)
            output_tensor = output_tensor.squeeze(0)
            loss = loss_fn(output_tensor, correct_tensor)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss   += loss.item()
            finished_process += 1
            finished_process_ += 1

            if loss.item() < lowest_loss:
                lowest_loss = loss.item()

            if loss.item() > highest_loss:
                highest_loss = loss.item()

            if loss.item() < 0.005 and correct_tensor.numpy().all() != 0:
                print(f"Loss: {loss.item():.4f}, Output: {output_tensor.detach().numpy()}, GT: {correct_tensor.numpy()}")

            if finished_process % opt.PRINT_RESULTS == 0:
                print_state(epoch, finished_process, running_loss, finished_process_, highest_loss, lowest_loss, optimizer.param_groups[0]['lr'])
                finished_process_, running_loss, highest_loss, lowest_loss = 0, 0, 0, 10000

            if finished_process % opt.SAVE_MODEL_AFTER == 0:
                save_model(model, str(finished_process))

        avg_loss_epoch = epoch_loss / dataset.size
        print(f"EOE {epoch}. Avg loss {avg_loss_epoch:.4f}")
        scheduler.step(avg_loss_epoch)
        epoch_loss = 0.0
        finished_process_, running_loss, highest_loss, lowest_loss = 0, 0, 0, 10000
        finished_process = 0
        epoch += 1


def print_state(epoch, finished_process, running_loss, finished_process_, highest_loss, lowest_loss, lr):
    loss_file = open(model_path_manager.loss_file, 'a+')
    summary_message = f"(epoch: {epoch}, finished_process: {finished_process}, lr: {lr}) a_loss: {running_loss/finished_process_:.3f} " \
                      f"h_loss: {highest_loss:.3f} l_loss: {lowest_loss:.3f}"
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

    train_model(model, CustomLoss())