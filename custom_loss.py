import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(CustomLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        differences = torch.abs(output - target) / (target + self.epsilon)
        weights = torch.tensor([40, 60], device=output.device)
        weighted_diff = differences * weights
        loss = weighted_diff.mean()
        return loss
