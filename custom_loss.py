import torch.nn as nn
import torch


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        open_output,  open_target  = output[0], target[0]
        high_output,  high_target  = output[1], target[1]
        low_output,   low_target   = output[2], target[2]
        close_output, close_target = output[3], target[3]
        vol_output,   vol_target   = output[4], target[4]

        close_difference = abs(close_output - close_target) / close_target
        open_difference  = abs(open_output  - open_target)  / open_target
        high_difference  = abs(high_output  - high_target)  / high_target
        low_difference   = abs(low_output   - low_target)   / low_target
        vol_difference   = abs(vol_output   - vol_target)   / vol_target

        loss = (500 * close_difference) + (300 * open_difference) + (80 * high_difference) + (80 * low_difference) + (40 * vol_difference)
        return loss.mean()
