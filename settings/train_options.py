import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cpu_threads', type=int, default=0, help="Number of cpu threads used by pytorch")


args, _ = parser.parse_known_args()