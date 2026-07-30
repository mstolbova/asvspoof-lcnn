import torch
from torch import nn


class ASVSpoofCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()

        if weight is not None:
            weight = torch.tensor(weight, dtype=torch.float32)

        self.loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor, **batch):
        labels = labels.long()
        return {"loss": self.loss(logits, labels)}
