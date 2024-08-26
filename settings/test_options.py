import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epoch', type=int, default=-1, help="Epoch of loading state")
parser.add_argument('--candle_input', type=int, default=100)
parser.add_argument('--correlation', action="store_true")
parser.add_argument('--graph', action="store_true")
parser.add_argument('--states', action="store_true")
parser.add_argument('--trend', action="store_true")
parser.add_argument('--data', type=str, required=False, default="", help="Name of data file for graph test")


args, _ = parser.parse_known_args()