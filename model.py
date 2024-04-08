import torch
import torch.nn as nn

"""
NN as input will take a matrix of 50 candles and predict next candle
Candle is a struct: [open_price, close_price, max_price, low_price, volume]
so matrix should look like this:

 n =    1            2            3            4              50
    open_price , open_price , open_price , open_price  ... open_price
    close_price, close_price, close_price, close_price ... close_price
    max_price  , max_price  , max_price  , max_price   ... max_price
    low_price  , low_price  , low_price  , low_price   ... low_price
    volume     , volume     , volume     , volume      ... volume
    
NN is going to output one candle [open_price, close_price, max_price, low_price, volume]
"""


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(5,  16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv1d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv1d(32, 48, 3, padding=1)
        self.lstm  = nn.LSTM(48, 64, batch_first=True)
        self.lin1  = nn.Linear(64 * 50, 4800)
        self.lin2  = nn.Linear(4800, 4800)
        self.lin3  = nn.Linear(4800, 2400)
        self.lin4  = nn.Linear(2400, 500)
        self.lin5  = nn.Linear(500 , 5)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = x.permute(0, 2, 1)
        x, (hn, cn) = self.lstm(x)
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = torch.relu(x)
        x = self.lin3(x)
        x = torch.relu(x)
        x = self.lin4(x)
        x = torch.relu(x)
        x = self.lin5(x)
        return x
