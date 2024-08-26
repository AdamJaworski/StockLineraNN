import os.path
import sys
import torch.nn as nn
from train.train import train_model
from settings.train_options import args
from launch_funcs import *
from settings import global_variables

def main():
    init_model_path_manager()
    load_model_file()

    from model.model import Model
    model = Model()

    set_device()

    global_variables.model_settings['candle_input'] = args.candle_input
    global_variables.model_settings['forward_candle_prediction'] = args.forward_candle_prediction
    global_variables.model_settings['lr'] = args.lr
    load_model_settings()

    state_name = ""
    if args.load_epoch != -1:
        for state in os.listdir(global_variables.model_path_manager.state_path):
            if f"EOE_{args.load_epoch}_" in state:
                state_name = state
                break
    else:
        state_name = "latest.pth"

    if args.continue_learning and os.path.exists(global_variables.model_path_manager.state_path / state_name):
        print(f"Loading saved state {state_name}...")
        model.load_state_dict(torch.load(global_variables.model_path_manager.state_path / state_name))

    try:
        train_model(model, nn.MSELoss())
    except KeyboardInterrupt:
        print("Stoping training....")
        sys.exit()


if __name__ == "__main__":
    main()