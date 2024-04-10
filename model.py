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
        self.conv1 = nn.Conv1d(5, 32, 3, padding=1)

        self.lstm1 = nn.LSTM(32, 64, batch_first=True, num_layers=3)

        self.lin1 = nn.Linear(64, 2)

        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        hidden_state = None

        for i in range(x.size(1)):  # Iterating over the sequence length
            candle = x[:, i, :].unsqueeze(1)  # Get the ith candle and add sequence dimension
            candle = candle.permute(0, 2, 1)
            candle = self.conv1(candle)
            candle = self.prelu1(candle)

            candle = candle.permute(0, 2, 1)  # Adjust for LSTM

            # Process the candle through the LSTM
            # LSTM input should be of shape (batch, seq_len, features), seq_len for a single candle is 1
            _, hidden_state = self.lstm1(candle, hidden_state)

        # After processing all candles, use the last hidden state to predict
        # Since we're interested in the final output, we use the last hidden state tuple (h, c)
        lstm_output = hidden_state[0].squeeze(0)  # Remove the sequence dimension
        x = self.prelu2(lstm_output)

        x = self.lin1(x)
        x = self.tanh(x)

        return x[0][0]
