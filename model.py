import torch
import torch.nn as nn
import torch.nn.functional as F
from train_options import opt


class Model(nn.Module):
    f"""
    NN as input will take a matrix of {opt.CANDLE_INPUT} candles and predict next candle
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
        self.lstm1 = nn.LSTM(5, 48, batch_first=True, num_layers=3)

        self.lin1 = nn.Linear(48, 2)
        self.lin2 = nn.Linear(3 * 2, 2)

        self.prelu1 = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        hidden_state = None

        for i in range(x.size(1)):
            candle = x[:, i, :].unsqueeze(1)
            _, hidden_state = self.lstm1(candle, hidden_state)

        lstm_output = hidden_state[0].squeeze(0)
        x = self.prelu1(lstm_output)

        x = self.lin1(x)
        x = torch.flatten(x)
        x = self.lin2(x)
        x = self.tanh(x)
        return x
