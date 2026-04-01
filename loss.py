import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1.):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, pred, target):
        # Add channel dimension to target
        target = target.unsqueeze(1)

        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = F.sigmoid(pred)
        pred = pred.contiguous()
        target = target.contiguous()

        intersection = (pred * target).sum(dim=2).sum(dim=2)

        dice = 1 - ((2. * intersection + self.smooth) / (
                    pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + self.smooth))

        loss = bce * self.bce_weight + dice.mean() * (1 - self.bce_weight)

        return loss


class TverskyLoss(nn.Module):
    #returns the Tversky loss per batch
    def __init__(self, smooth=1e-10, alpha=0.5, beta=0.5):
        super().__init__()
        self.smooth = smooth
        self.alpha = alpha
        self.beta = beta

    def forward(self, y_pred, y_true):
        # Flatten both prediction and GT tensors
        y_pred_flat = torch.flatten(y_pred)
        y_true_flat = torch.flatten(y_true)
        # calculate the number of true positives, false positives and false negatives
        tp = (y_pred_flat * y_true_flat).sum()
        fp = (y_pred_flat * (1 - y_true_flat)).sum()
        fn = ((1 - y_pred_flat) * y_true_flat).sum()
        # calculate the Tversky index
        tversky = tp/(tp + self.alpha * fn + self.beta * fp + self.smooth)
        # return the loss
        return 1 - tversky
