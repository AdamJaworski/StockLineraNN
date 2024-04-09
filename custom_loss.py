import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(CustomLoss, self).__init__()

    def forward(self, output, target):
        differences = 100 * torch.abs(target - output)
        weights = torch.tensor([0.4, 0.6], device=output.device)
        weighted_diff = differences * weights
        loss = weighted_diff.mean()
        return loss
