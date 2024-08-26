import time
import json
import pathlib
import random
from torch import save
from torch.optim.lr_scheduler import ReduceLROnPlateau
from settings import global_variables
from settings.train_options import args
from settings.common_options import common_args

def train_model(model, loss_fn):
    model.to(global_variables.device)
    optimizer  = global_variables.TRAIN_MODELS[args.train_model.lower()](params=model.parameters(), lr=global_variables.model_settings['lr'])

    creating_dataset_time_start = time.time()
    dataset    = global_variables.DATASETS[args.dataset.lower()](global_variables.TRAIN_DATA_PATH)
    print(f"Creating dataset took: {time.time() - creating_dataset_time_start:.3f} s")


    scheduler         = ReduceLROnPlateau(optimizer, 'min', factor=args.lr_drop_off_factor, patience=args.drop_off_patience)

    epoch             = global_variables.model_settings['epoch']
    finished_process  = global_variables.model_settings['finished_process']

    epoch_loss        = 0.0
    highest_loss      = 0
    lowest_loss       = 10000
    finished_process_in_epoch = 0

    interation        = [*range(dataset.get_size())]
    print(f"Size of dataset: {dataset.get_size()}/{dataset.total_size / 1024:.2f}KB")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Training {num_trainable_params} parameters")

    while True:
        #print(f"Starting epoch: {epoch}")
        random.shuffle(interation)

        last_print = time.time()
        for i in interation:
            if common_args.debug:
                iteration_start = time.time()

            optimizer.zero_grad()

            # Get new data
            input_tensor, correct_tensor = dataset.get(i)

            input_tensor = input_tensor.unsqueeze(0)
            output_tensor = model(input_tensor)

            output_tensor = output_tensor.squeeze(0)
            loss = loss_fn(output_tensor, correct_tensor)
            #print(f"{i}. Loss: {loss.item()}, Output: {output_tensor}, Correct: {correct_tensor} \r", end="")

            loss.backward()
            optimizer.step()

            epoch_loss   += loss.item()
            finished_process += 1
            finished_process_in_epoch += 1

            if loss.item() < lowest_loss:
                lowest_loss = loss.item()

            if loss.item() > highest_loss:
                highest_loss = loss.item()

            if common_args.debug:
                print(f"Time for iteration: {time.time() - iteration_start} s")

            if finished_process_in_epoch == args.epoch_size:
                if common_args.debug:
                    print(f"Time for state run: {(time.time() - last_print):.2f} s")

                epoch_loss = epoch_loss / args.epoch_size # / dataset.get_size() TODO
                save_model(model, f"EOE_{epoch}_{epoch_loss:.8f}")
                print_state(epoch, finished_process, epoch_loss, highest_loss, lowest_loss, optimizer.param_groups[0]['lr'])
                scheduler.step(epoch_loss)
                epoch += 1

                global_variables.model_settings['epoch']              = epoch
                global_variables.model_settings['lr']                 = optimizer.param_groups[0]['lr']
                global_variables.model_settings['finished_process']  += finished_process_in_epoch
                with open(f"{global_variables.model_path_manager.settings_file}", "w") as file:
                    json.dump(global_variables.model_settings, file, indent=4)

                finished_process_in_epoch, running_loss, highest_loss, lowest_loss, epoch_loss = 0, 0, 0, 10000, 0


def print_state(epoch, finished_process, epoch_loss, highest_loss, lowest_loss, lr):
    loss_file = open(global_variables.model_path_manager.loss_file, 'a+')
    summary_message = f"(epoch: {epoch}, finished_process: {finished_process}, lr: {lr}) a_loss: {epoch_loss:.8f} " \
                      f"h_loss: {highest_loss:.8f} l_loss: {lowest_loss:.8f}"
    print(summary_message)
    loss_file.write(summary_message + '\n')
    loss_file.close()


def save_model(model_instance, name_of_save: str):
    model_instance.to('cpu')

    # Current save
    if pathlib.Path.exists(global_variables.model_path_manager.state_path / (name_of_save + '.pth')):
        pathlib.Path.unlink(global_variables.model_path_manager.state_path / (name_of_save + '.pth'))
    save(model_instance.state_dict(), global_variables.model_path_manager.state_path / (name_of_save + '.pth'))

    # Latest save
    if pathlib.Path.exists(global_variables.model_path_manager.state_path / 'latest.pth'):
        pathlib.Path.unlink(global_variables.model_path_manager.state_path / 'latest.pth')

    save(model_instance.state_dict(), global_variables.model_path_manager.state_path / 'latest.pth')

    model_instance.to(global_variables.device)
