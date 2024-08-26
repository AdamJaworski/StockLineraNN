from dataset.open_close import OpenClose
from dataset.trend import Trend
from torch.optim import Adam, SGD, AdamW
from pathlib import Path
import torch

DATASETS = {
    "open_close": OpenClose,
    "trend": Trend
}


TRAIN_MODELS = {
    'adam': Adam,
    'adamw': AdamW,
    'sgd': SGD
}


MAIN_PATH = Path('.').resolve()
TRAIN_DATA_PATH = MAIN_PATH / 'Data/DataTraining'
TEST_DATA_PATH = MAIN_PATH / 'Data/DataTest'
MODELS_PATH = MAIN_PATH / 'models'

TENSOR_DATA_TYPE = torch.float32

device = None
model_path_manager = None

model_settings = {
    'epoch': 0,
    'lr': 0.01,
    'candle_input': 100,
    'finished_process': 0,
    'forward_candle_prediction': 10
}