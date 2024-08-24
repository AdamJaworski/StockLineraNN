"""
No best version - all bad , dataset too small?
"""

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

        self.lin1 = nn.Linear(500, 3000)
        self.lin2 = nn.Linear(3000, 600)
        self.lin3 = nn.Linear(600, 200)
        self.lin4 = nn.Linear(200, 2)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.sigmoid(self.lin1(x), 60)
        x = self.sigmoid(self.lin2(x), 50)
        x = self.sigmoid(self.lin3(x), 100)
        x = self.lin4(x)
        return self.sigmoid(x, 10)


    def sigmoid(self, x, factor):
        return 2 * (1 / (1 + torch.exp(-x/factor))) -1