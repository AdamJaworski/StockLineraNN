from PathManager import PathManager
import torch

# import argparse

class Options:
    SAVE_MODEL_AFTER: int
    PRINT_RESULTS: int
    CONTINUE_LEARNING: bool
    MODEL: str
    LR: float
    LR_DROP_OFF_FACTOR: float
    STARTING_EPOCH: int
    CANDLE_INPUT: int
    PATH: PathManager
    TRAIN_MODEL: torch.optim.Optimizer
    device: torch.device
    DEBUG: bool
    LOAD_MODEL: str
    DROP_OFF_PATIENCE: int

    def __init__(self):
        pass

    def to_dict(self) -> dict:
        return {
            'SAVE_MODEL_AFTER':   self.SAVE_MODEL_AFTER,
            'PRINT_RESULTS':      self.PRINT_RESULTS,
            'CONTINUE_LEARNING':  self.CONTINUE_LEARNING,
            'DEBUG':              self.DEBUG,
            'LR':                 self.LR,
            'LR_DROP_OFF_FACTOR': self.LR_DROP_OFF_FACTOR,
            'STARTING_EPOCH':     self.STARTING_EPOCH,
            'CANDLE_INPUT':       self.CANDLE_INPUT,
            'LOAD_MODEL':         self.LOAD_MODEL
        }

    def load_settings(self, data: dict):
        self.SAVE_MODEL_AFTER       = data['SAVE_MODEL_AFTER']
        self.PRINT_RESULTS          = data['PRINT_RESULTS']
        self.CONTINUE_LEARNING      = data['CONTINUE_LEARNING']
        self.DEBUG                  = data['DEBUG']
        self.LR                     = data['LR']
        self.LR_DROP_OFF_FACTOR     = data['LR_DROP_OFF_FACTOR']
        self.STARTING_EPOCH         = data['STARTING_EPOCH']
        self.CANDLE_INPUT           = data['CANDLE_INPUT']
        self.LOAD_MODEL             = data['LOAD_MODEL']



opt = Options()

def add_argument(**kwargs):
    for key, value in kwargs.items():
        setattr(opt, key, value)


add_argument(SAVE_MODEL_AFTER=-1) # if < 0 saves model after each epoch, if 0 doesn't save model at all
add_argument(PRINT_RESULTS=-1)
add_argument(CONTINUE_LEARNING=False)
add_argument(DEBUG=True)
add_argument(MODEL='sigmoid_no_relu_v3')
add_argument(LR=0.01)
add_argument(LR_DROP_OFF_FACTOR=0.5)
add_argument(DROP_OFF_PATIENCE=3)
add_argument(STARTING_EPOCH=0)
add_argument(CANDLE_INPUT=100)
path_manager = PathManager(opt.MODEL)
add_argument(PATH=path_manager)
add_argument(TRAIN_MODEL=torch.optim.Adam)
add_argument(LOAD_MODEL='latest.pth')