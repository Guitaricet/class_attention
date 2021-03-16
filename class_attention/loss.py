import torch
import torch.nn as nn
import torch.nn.functional as F


# based on
# https://github.com/pytorch/pytorch/issues/7455#issuecomment-568147262
class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, preds, target):
        log_prob = F.log_softmax(preds, dim=-1)
        with torch.no_grad():
            smooth_target = preds.new_ones(preds.size()) * self.smoothing / (preds.size(-1) - 1.)
            smooth_target.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))

        loss = (-smooth_target * log_prob).sum(dim=-1).mean()
        return loss
