import numpy as np
import pandas as pd
import torch_directml
from numba import njit

def get_rsi_for_data(data, window=14):
    assert len(data) == window, "Wrong data size to calculate rsi"

    try:
        delta = np.diff(data)
    except Exception as e:
        print(data)
        raise e
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.average(gain)
    avg_loss = np.average(loss)

    rs = avg_gain / avg_loss if avg_loss != 0 else 0
    rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100

    return rsi

def normalize_data(data):
    for i in reversed(range(len(data))):
        if i == 0:
            continue
        reference_candle = data[i - 1]
        candle_to_normalize = data[i]
        new_candle = (candle_to_normalize - reference_candle[3]) / reference_candle[3]
        new_candle = new_candle[:len(new_candle) - 1]
        new_candle = np.append(new_candle, candle_to_normalize[len(candle_to_normalize) - 1])
        data[i] = new_candle.astype(np.float16)

    reference_price = data[0][3]
    data = data[1:]
    return reference_price, data


def is_directml_available():
    try:
        dml = torch_directml.device()
        return True
    except Exception as e:
        return False
