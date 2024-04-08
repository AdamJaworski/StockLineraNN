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
        self.conv1      = nn.Conv1d(6,  24, 3, padding=1)
        self.conv2      = nn.Conv1d(24, 32, 3, padding=1)
        self.drop2      = nn.Dropout(0.3)

        self.lstm1      = nn.LSTM(32, 128, batch_first=True)
        self.lstm2      = nn.LSTM(128, 128, batch_first=True, dropout=0.2)
        self.lstm3      = nn.LSTM(128, 64, batch_first=True)
        self.lstm_drop1 = nn.Dropout(0.3)

        self.lin1       = nn.Linear(64 * 150, 2400)
        self.lin_drop1  = nn.Dropout(0.4)
        self.lin2       = nn.Linear(2400, 200)
        self.lin_drop2  = nn.Dropout(0.4)
        self.lin3       = nn.Linear(200, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop2(x)

        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm1(x)
        x, (hn, cn) = self.lstm2(x)
        x, (hn, cn) = self.lstm3(x)
        x = self.lstm_drop1(x)
        x = x.reshape(x.shape[0], -1)  # Flatten

        x = F.relu(self.lin1(x))
        x = self.lin_drop1(x)
        x = F.relu(self.lin2(x))
        x = self.lin_drop2(x)
        x = self.lin3(x)
        return x
