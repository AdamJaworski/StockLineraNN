import json
import os
import torch
import torch.nn as nn
from settings.global_options import opt
from datetime import datetime
from train.train import train_model
from settings.train_options import args
import shutil

def main():
    if os.path.exists(opt.PATH.model_file):
        print("Loading model file")
        os.remove('./model/model.py')
        shutil.copy(opt.PATH.model_file, './model/model.py')
    else:
        shutil.copy('./model/model.py', opt.PATH.model_file)

    from model.model import Model
    model = Model()

    if os.path.exists(opt.PATH.settings_file):
        print("Found settings file")
        with open(f"{opt.PATH.settings_file}", "r") as file:
            opt.load_settings(json.load(file))
    else:
        with open(f"{opt.PATH.settings_file}", "w") as file:
            json.dump(opt.to_dict(), file, indent=4)

    device_name = ''
    if torch.cuda.is_available():
        opt.device = torch.device('cuda')
        device_name = torch.cuda.get_device_name(opt.device)

    # elif is_directml_available():
    #     opt.device = torch_directml.device()
    #     device_name = torch_directml.device_name(opt.device.index)
    #     print("Dropping to directml")

    else:
        opt.device = torch.device('cpu')
        device_name = 'CPU'
        if args.cpu_threads > 0:
            torch.set_num_threads(args.cpu_threads)

    print(f"Using {device_name} as main device")
        
    if opt.DEBUG:
        torch.autograd.set_detect_anomaly(True)

    if opt.CONTINUE_LEARNING:
        try:
            print(f"Loading saved state {opt.LOAD_MODEL}...")
            model.load_state_dict(torch.load(opt.PATH.state_path / opt.LOAD_MODEL))
        except Exception as e:
            raise e
    else:
        for module in model.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # Implement LSTM bias initialization for forget gate
                        n = param.size(0)
                        start, end = n // 4, n // 2
                        param.data[start:end].fill_(1.)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    


    loss_file = open(opt.PATH.loss_file, 'a+')
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    start_message = '=' * 55 + f'{dt_string}' + '=' * 55
    loss_file.write(start_message + '\n')
    loss_file.close()

    train_model(model, nn.MSELoss(), opt)


if __name__ == "__main__":
    main()