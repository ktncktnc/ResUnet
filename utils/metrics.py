from typing import Optional, Dict, Any, Callable

import torch
import numpy as np
import torchmetrics
from torch import nn, Tensor
from torchmetrics.classification.stat_scores import StatScores
from torchmetrics.functional.classification.stat_scores import _reduce_stat_scores, _stat_scores_update
from torchmetrics.utilities.enums import AverageMethod, MDMCAverageMethod


def _dice_compute(
    tp: Tensor,
    fp: Tensor,
    fn: Tensor,
    average: str,
    mdmc_average: Optional[str],
    zero_division: int = 0,
) -> Tensor:
    numerator = 2 * tp
    denominator = 2 * tp + fp + fn

    if average == AverageMethod.MACRO and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        cond = tp + fp + fn == 0
        numerator = numerator[~cond]
        denominator = denominator[~cond]

    if average == AverageMethod.NONE and mdmc_average != MDMCAverageMethod.SAMPLEWISE:
        # a class is not present if there exists no TPs, no FPs, and no FNs
        meaningless_indeces = torch.nonzero((tp | fn | fp) == 0).cpu()
        numerator[meaningless_indeces, ...] = -1
        denominator[meaningless_indeces, ...] = -1

    return _reduce_stat_scores(
        numerator=numerator,
        denominator=denominator,
        weights=None if average != "weighted" else tp + fn,
        average=average,
        mdmc_average=mdmc_average,
        zero_division=zero_division,
    )


class Dice(StatScores):
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(
        self,
        zero_division: int = 0,
        num_classes: Optional[int] = None,
        threshold: float = 0.5,
        average: str = "micro",
        mdmc_average: Optional[str] = "global",
        ignore_index: Optional[int] = None,
        top_k: Optional[int] = None,
        multiclass: Optional[bool] = None,
        **kwargs: Dict[str, Any],
    ) -> None:
        allowed_average = ("micro", "macro", "weighted", "samples", "none", None)
        if average not in allowed_average:
            raise ValueError(f"The `average` has to be one of {allowed_average}, got {average}.")

        super().__init__(
            reduce="macro" if average in ("weighted", "none", None) else average,
            mdmc_reduce=mdmc_average,
            threshold=threshold,
            top_k=top_k,
            num_classes=num_classes,
            multiclass=multiclass,
            ignore_index=ignore_index,
            **kwargs,
        )

        self.average = average
        self.zero_division = zero_division

    def update(self, dice_preds: Tensor, dice_target: Tensor) -> None:  # type: ignore
        tp, fp, tn, fn = _stat_scores_update(
            dice_preds,
            dice_target,
            reduce=self.reduce,
            mdmc_reduce=self.mdmc_reduce,
            threshold=self.threshold,
            num_classes=self.num_classes,
            top_k=self.top_k,
            multiclass=self.multiclass,
            ignore_index=self.ignore_index,
        )

        # Update states
        if self.reduce != AverageMethod.SAMPLES and self.mdmc_reduce != MDMCAverageMethod.SAMPLEWISE:
            self.tp += tp
            self.fp += fp
            self.tn += tn
            self.fn += fn
        else:
            self.tp.append(tp)
            self.fp.append(fp)
            self.tn.append(tn)
            self.fn.append(fn)

    def compute(self) -> Tensor:
        """Computes the dice score based on inputs passed in to ``update`` previously.
        Return:
            The shape of the returned tensor depends on the ``average`` parameter:
            - If ``average in ['micro', 'macro', 'weighted', 'samples']``, a one-element tensor will be returned
            - If ``average in ['none', None]``, the shape will be ``(C,)``, where ``C`` stands  for the number
              of classes
        """
        tp, fp, _, fn = self._get_final_stats()
        return _dice_compute(tp, fp, fn, self.average, self.mdmc_reduce, self.zero_division)


class TrackingMetric(torchmetrics.Metric):
    def __init__(
        self,
        name: str,
        average: Optional[str] = "micro",
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(**kwargs)
        self.name = name
        self.average = average
        if average not in ["micro"]:
            raise ValueError(f"The `reduce` {average} is not valid.")

        self.add_state("values", default=[], dist_reduce_fx="cat")

    def update(self, value: Dict) -> None:
        if self.name not in value.keys():
            raise ValueError("Passed dict doesn't contain key {name}".format(name=self.name))

        self.values.append(value[self.name])

    def compute(self) -> Tensor:
        return torch.mean(torch.tensor(self.values))


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
    num_in_target = y_pred.size(0)
    y_true = y_true.cpu().numpy()
    pred = torch.argmax(y_pred, 1).cpu().numpy()
    return np.sum(pred == y_true)/num_in_target
