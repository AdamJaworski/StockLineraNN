import numpy as np
import pandas as pd


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