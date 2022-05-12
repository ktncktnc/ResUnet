import torch
import numpy as np
from torch import nn


class BCEDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super().__init__()

        if weight is None:
            weight = [1.0, 1.0]
        self.weight = weight

    def forward(self, input, target):
        pred = torch.reshape(input, (-1,))
        truth = torch.reshape(target, (-1,))

        # BCE loss
        bce_loss = nn.BCELoss()(pred, truth).double()

        # Dice Loss
        dice_coef = (2.0 * (pred * truth).double().sum() + 1) / (
            pred.double().sum() + truth.double().sum() + 1
        )

        return self.weight[0]*bce_loss + self.weight[1]*(1 - dice_coef)


# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class MetricTracker(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# https://stackoverflow.com/questions/48260415/pytorch-how-to-compute-iou-jaccard-index-for-semantic-segmentation
def jaccard_index(input, target):

    intersection = (input * target).long().sum().data.cpu()[0]
    union = (
        input.long().sum().data.cpu()[0]
        + target.long().sum().data.cpu()[0]
        - intersection
    )

    if union == 0:
        return float("nan")
    else:
        return float(intersection) / float(max(union, 1))


# https://github.com/pytorch/pytorch/issues/1249
def dice_coeff(input, target):
    num_in_target = input.size(0)

    smooth = 1.0

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    loss = (2.0 * intersection + smooth) / (pred.sum(1) + truth.sum(1) + smooth)

    return loss.mean().item()

def np_dice_coeff(input, target):
    num_in_target = input.shape[0]
    smooth = 1.0

    pred = np.reshape(input, (num_in_target, -1))
    truth = np.reshape(target, (num_in_target, -1))

    intersection = np.multiply(pred, truth).sum(1)
    loss = (2.0 * intersection + smooth)/(pred.sum(1) + truth.sum(1) + smooth)

    return loss.sum()/num_in_target


def np_metrics(y_pred, y_true):
    num_in_target = y_pred.shape[0]
    y_pred = np.reshape(y_pred, (num_in_target, -1))
    y_true = np.reshape(y_true, (num_in_target, -1))

    tp = (y_pred* y_true).sum(1)
    tn = ((1 - y_true) * (1 - y_pred)).sum(1)
    fp = ((1 - y_true) * y_pred).sum(1)
    fn = (y_true * (1 - y_pred)).sum(1)

    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    dice = 2*tp/(2*tp + fp + fn)

    return precision.mean(), recall.mean(), f1.mean(), dice.mean()

def acc(y_pred, y_true):
    num_in_target = input.size(0)

    smooth = 1.0

    y_true = y_true.cpu().numpy()
    pred = torch.argmax(y_pred, 1).cpu().numpy()
    return np.sum(pred == y_true)/num_in_target
