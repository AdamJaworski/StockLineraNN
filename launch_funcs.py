import json
import torch
import os
import shutil
import torch_directml
from settings import global_variables
from settings.common_options import common_args
from model.model_paths import ModelPaths

def init_model_path_manager():
    global_variables.model_path_manager = ModelPaths(common_args.model)

def set_device():
    """
    sets global_variables.device based on user args input
    """
    if common_args.directml:
        global_variables.device = torch_directml.device()
        device_name = torch_directml.device_name(global_variables.device.index)
    else:
        if torch.cuda.is_available():
            global_variables.device = torch.device('cuda')
            device_name = torch.cuda.get_device_name(global_variables.device)
        else:
            global_variables.device = torch.device('cpu')
            if common_args.cpu_threads > 0:
                torch.set_num_threads(common_args.cpu_threads)
            device_name = 'CPU'

    print(f"Using {device_name} as main device")


def load_model_file():
    if os.path.exists(global_variables.model_path_manager.model_file):
        print("Loading model.py")
        os.remove('./model/model.py')
        shutil.copy(global_variables.model_path_manager.model_file, './model/model.py')
    else:
        shutil.copy('./model/model.py',global_variables.model_path_manager.model_file)

def load_model_settings():
    if common_args.load_settings:
        if os.path.exists(global_variables.model_path_manager.settings_file):
            with open(f"{global_variables.model_path_manager.settings_file}", "r") as file:
                print("Loading model settings")
                global_variables.model_settings = json.load(file)
        else:
            with open(f"{global_variables.model_path_manager.settings_file}", "w") as file:
                json.dump(global_variables.model_settings, file)

def check_debug():
    if common_args.debug:
        torch.autograd.set_detect_anomaly(True)