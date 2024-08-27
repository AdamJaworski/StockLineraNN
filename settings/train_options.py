import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, required=True, help="Name of dataset for training")

parser.add_argument('--continue_learning', type=bool, default=True)
parser.add_argument('--lr_drop_off_factor', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--drop_off_patience', type=int, default=25)
parser.add_argument('--train_model', type=str, default='adamw')
parser.add_argument('--candle_input', type=int, default=100)
parser.add_argument('--forward_candle_prediction', type=int, default=10)
parser.add_argument('--epoch_size', type=int, default=50e3)

parser.add_argument('--load_epoch', type=int, default=-1)
args, _ = parser.parse_known_args()