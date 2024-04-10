class Options:
    def __init__(self):
        pass


opt = Options()


def add_argument(**kwargs):
    for key, value in kwargs.items():
        setattr(opt, key, value)


add_argument(SAVE_MODEL_AFTER=1000)
add_argument(PRINT_RESULTS=1000)
add_argument(CONTINUE_LEARNING=False)
add_argument(LOAD_SETTINGS=True)    # TODO
add_argument(MODEL='Gamma')
add_argument(LR=1e-2)
add_argument(LR_DROPOFF_FACTOR=0.5)
add_argument(STARTING_EPOCH=0)
add_argument(CANDLE_INPUT=100)
