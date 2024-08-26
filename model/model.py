"""

"""
import sys

import torch
import torch.nn as nn

class Model(nn.Module):
    f"""
    NN as input will take a matrix of 100 candles and predict next candle
    Candle is a struct: [open_price, close_price, max_price, low_price, RSI]
    so matrix should look like this:

     n =    1            2            3            4              opt.CANDLE_INPUT
        open_price , open_price , open_price , open_price  ... open_price
        close_price, close_price, close_price, close_price ... close_price
        max_price  , max_price  , max_price  , max_price   ... max_price
        low_price  , low_price  , low_price  , low_price   ... low_price
        RSI        , RSI        , RSI        , RSI         ... RSI

    NN is going to output open_price, close_price
    """

    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv1d(100, 600, 5)

        self.lin1 = nn.Linear(500, 1800)
        self.lin2 = nn.Linear(1800, 600)
        self.lin3 = nn.Linear(600, 3)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.sigmoid(torch.flatten(y), 150)

        x = torch.flatten(x)
        x = self.lin1(x)
        x = self.sigmoid(self.lin2(x), 150)

        x = self.lin3(x + y)
        return self.softmax(x)

    def sigmoid(self, x, factor):
        """
        changes x into values between -1, 1
        """
        return 2 * (1 / (1 + torch.exp(-x/factor))) -1