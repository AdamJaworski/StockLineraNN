import sys
import time
import json
from torch import save
import torch.optim as optim
from dataset.dataset_training import DatasetTraining
import pathlib
import random


def train_model(model, loss_fn, opt):
    model.to(opt.device)
    optimizer         = opt.TRAIN_MODEL(params=model.parameters(), lr=opt.LR)
    scheduler         = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=opt.LR_DROP_OFF_FACTOR, patience=opt.DROP_OFF_PATIENCE)
    dataset           = DatasetTraining(opt.PATH.data_train)
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
            input_tensor = input_tensor.to(opt.device)
            correct_tensor = correct_tensor.to(opt.device)
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

            if loss.item() < 0.0000000050 and correct_tensor.numpy().all() != 0:
                print(f"Loss: {loss.item():.10f}, Output: {output_tensor.detach().numpy()}, GT: {correct_tensor.numpy()}")

            if finished_process % opt.PRINT_RESULTS == 0 and opt.PRINT_RESULTS > 0:
                print_state(epoch, finished_process, running_loss, finished_process_, highest_loss, lowest_loss, optimizer.param_groups[0]['lr'], opt)
                finished_process_, running_loss, highest_loss, lowest_loss = 0, 0, 0, 10000

            if finished_process % opt.SAVE_MODEL_AFTER == 0 and opt.SAVE_MODEL_AFTER > 0:
                save_model(model, str(finished_process), opt)

        avg_loss_epoch = epoch_loss / dataset.size

        if opt.SAVE_MODEL_AFTER < 1:
            save_model(model, f"EOE_{epoch}_{avg_loss_epoch:.8f}", opt)

        if opt.PRINT_RESULTS > 0:
            print(f"EOE {epoch}. Avg loss {avg_loss_epoch:.8f}")
        else:
            print_state(epoch, finished_process, running_loss, finished_process_, highest_loss, lowest_loss,
                        optimizer.param_groups[0]['lr'], opt)

        scheduler.step(avg_loss_epoch)
        epoch_loss = 0.0
        finished_process_, running_loss, highest_loss, lowest_loss = 0, 0, 0, 10000
        finished_process = 0
        epoch += 1

        opt.CONTINUE_LEARNING = True
        opt.STARTING_EPOCH = epoch
        opt.LR = optimizer.param_groups[0]['lr']

        with open(f"{opt.PATH.settings_file}", "w") as file:
            json.dump(opt.to_dict(), file, indent=4)

last_print = time.time()

def print_state(epoch, finished_process, running_loss, finished_process_, highest_loss, lowest_loss, lr, opt):

    if opt.DEBUG:
        global last_print
        print(f"Time for state run: {(time.time() - last_print):.2f} s")
        last_print = time.time()

    loss_file = open(opt.PATH.loss_file, 'a+')
    summary_message = f"(epoch: {epoch}, finished_process: {finished_process}, lr: {lr}) a_loss: {running_loss/finished_process_:.8f} " \
                      f"h_loss: {highest_loss:.8f} l_loss: {lowest_loss:.8f}"
    print(summary_message)
    loss_file.write(summary_message + '\n')
    loss_file.close()


def save_model(model_instance, name_of_save: str, opt):
    model_instance.to('cpu')
    # Current save
    if pathlib.Path.exists(opt.PATH.state_path / (name_of_save + '.pth')):
        pathlib.Path.unlink(opt.PATH.state_path / (name_of_save + '.pth'))
    save(model_instance.state_dict(), opt.PATH.state_path / (name_of_save + '.pth'))

    # Latest save
    if pathlib.Path.exists(opt.PATH.state_path / 'latest.pth'):
        pathlib.Path.unlink(opt.PATH.state_path / 'latest.pth')
    save(model_instance.state_dict(), opt.PATH.state_path / 'latest.pth')
    model_instance.to(opt.device)
