import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cpu_threads', type=int, default=0, help="Number of cpu threads used by pytorch")
parser.add_argument('--directml', action="store_true")

parser.add_argument('--model', type=str, required=True, help="Name of model to test")

parser.add_argument('--debug', action="store_true")
parser.add_argument('--load_settings', type=bool, default=True)

common_args, _ = parser.parse_known_args()
