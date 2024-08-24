import numpy as np
import pandas as pd
import torch_directml

def calculate_rsi(data, window=14):
    if len(data) < 6:
        return -1
    if len(data) < window + 1:
        window = len(data)

    delta = np.diff(data)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = np.average(gain[-window:])  # Calculate average gain for last 'window' periods
    avg_loss = np.average(loss[-window:])  # Calculate average loss for last 'window' periods

    rs = avg_gain / avg_loss if avg_loss != 0 else 0  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs)) if avg_loss != 0 else 100  # RSI is 100 if avg_loss is 0

    return rsi


def get_rsi(data, data_set=True) -> list:
    if data_set:
        close_prices = np.array([candle[3] for candle in data])
    else:
        close_prices = data

    rsi_values = []
    for i in range(len(close_prices)):
        rsi_values.append(calculate_rsi(close_prices[:i + 1]))

    i = 0
    while rsi_values[i] == -1:
        i += 1

    y = 0
    while rsi_values[y] == -1:
        rsi_values[y] = rsi_values[i]
        y += 1

    return rsi_values


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
