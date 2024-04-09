import torch
import torch.nn as nn
import torch.nn.functional as F

"""
NN as input will take a matrix of 150 candles and predict next candle
Candle is a struct: [open_price, close_price, max_price, low_price, volume, RSI]
so matrix should look like this:

 n =    1            2            3            4              50
    open_price , open_price , open_price , open_price  ... open_price
    close_price, close_price, close_price, close_price ... close_price
    max_price  , max_price  , max_price  , max_price   ... max_price
    low_price  , low_price  , low_price  , low_price   ... low_price
    volume     , volume     , volume     , volume      ... volume
    RSI        , RSI        , RSI        , RSI         ... RSI
    
NN is going to output open_price, close_price
"""


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1      = nn.Conv1d(5,  32, 3, padding=1)
        self.batch1     = nn.BatchNorm1d(32)

        self.lstm1      = nn.LSTM(32, 64, batch_first=False, num_layers=2)

        self.lin1       = nn.Linear(150 * 64, 1200)
        self.lin_drop1  = nn.Dropout(0.4)
        self.lin2       = nn.Linear(1200, 2)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.prelu1(x)

        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm1(x)
        x = self.prelu2(x)

        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.lin1(x)
        x = self.lin_drop1(x)
        x = self.prelu3(x)
        x = self.lin2(x)
        x = self.tanh(x)
        return x
