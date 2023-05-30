from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss


class LossCalculator(ABC):

    @abstractmethod
    def calculate_loss(self, outputs, sample):
        pass


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets, mask=None):
        bce_loss = F.binary_cross_entropy_with_logits(preds, targets, reduction='none')
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
               (1. - probas) ** self.gamma * bce_loss + \
               (1. - targets) * probas ** self.gamma * bce_loss
        if mask is not None:
            loss = loss * mask
        #loss = loss.mean()
        return loss


class BCEFocal2WayLoss(nn.Module):
    def __init__(self, weights=[1, 1]):
        super().__init__()
        self.focal = BCEFocalLoss()
        self.weights = weights
        print(f"init loss BCE2W... with weights {weights}")
    ## TODO: penalize rating
    ## TODO: fix label smoothing
    def forward(self, input, target, mask=None):
        input_ = input["logit"]
        w      = target['weight'].float().to(input_.device)
        target = target["labels"].float().to(input_.device).clamp(0.01, 0.99)
        
        framewise_output = input["framewise_logit"]
        clipwise_output_with_max, _ = framewise_output.max(dim=1)
        loss = self.focal(input_, target, mask)
        aux_loss = self.focal(clipwise_output_with_max, target, mask)
        loss_ = self.weights[0] * loss + self.weights[1] * aux_loss
        
        ## rating penalty
        loss_ = (loss_.sum(dim=1) * w) / w.sum()
        loss_ = loss_.sum()
        return loss_

class BCEF2WLossCalculator(LossCalculator):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = BCEFocal2WayLoss(**kwargs)
        print("init loss BCE2W...")

    def calculate_loss(self, outputs, targets):
        return self.loss(outputs, targets)


class BCEFocal1WayLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = BCEFocalLoss()

    def forward(self, input, target, mask=None):
        input_ = input["logit"]
        w = target['weight'].float().to(input_.device)
        target = target["labels"].float().to(input_.device).clamp(0.01, 0.99)
        loss = self.focal(input_, target, mask)
        loss_ = (loss.sum(dim=1) * w) / w.sum()
        loss_ = loss_.sum()
        return loss_


class BCEF1WLossCalculator(LossCalculator):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = BCEFocal1WayLoss(**kwargs)
        print("init loss BCE1W...")

    def calculate_loss(self, outputs, targets):
        return self.loss(outputs, targets)


class BCEBirdLossCalculator(LossCalculator):
    def __init__(self, pos_weight = None, **kwargs):
        super().__init__()
        if pos_weight is not None:
            pos_weight = np.array([row[1] for row in pos_weight])
            pos_weight = torch.from_numpy(pos_weight).float().cuda()
        self.loss = BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        print("init loss BCEBirdLoss...")

    def calculate_loss(self, outputs, targets):
        input = outputs["logit"]
        target = targets["labels"].float().cuda()
        loss = self.loss(input, target)
        #sum by class, mean by sample
        return loss.sum(dim=1).mean()

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SmoothingBirdLossCalculator(LossCalculator):
    def __init__(self, **kwargs):
        super().__init__()
        self.loss = LabelSmoothing()
        print("init loss BCEF...")

    def calculate_loss(self, outputs, targets):
        input = outputs["logit"]
        target = targets["labels"].float().cuda()
        loss = self.loss(input, target)
        #sum by class, mean by sample
        return loss.sum(dim=1).mean()


def tp_score(y_preds, y_true, eps=1e-3):
    #TP Score = (TP)/(TP+FP)
    tp = torch.sum(y_preds * y_true, dim=0)
    tp_fp = torch.sum(y_preds, dim=0) + eps
    return (tp / tp_fp).mean()


def tn_score(y_preds, y_true, eps=1e-3):
    #TN Score = (TN)/(TN+FN)
    tn = torch.sum((1 - y_preds) * (1 - y_true), dim=0)
    tn_fn = torch.sum(1 - y_preds, dim=0)
    return (tn / (tn_fn + eps)).mean()


class LBLossCalculator(LossCalculator):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, outputs, targets):
        input = outputs["logit"]
        target = targets["labels"].float().cuda()

        probs = torch.sigmoid(input)
        return 1 - (tp_score(probs, target) + tn_score(probs, target))/2


def sensitivity(y_preds, y_true, eps=1e-3):
    tp = torch.sum(y_preds * y_true, dim=0)
    tp_fp = torch.sum(y_preds, dim=0) + eps
    return (tp / tp_fp).mean()


def specificity(y_preds, y_true, eps=1e-3):
    tn = torch.sum((1 - y_preds) * (1 - y_true), dim=0)
    fp = torch.sum(y_preds * (1 - y_true), dim=0)
    return (tn / (tn + fp + eps)).mean()


class SensitivitySpecificityLossCalculator(LossCalculator):
    def __init__(self):
        super().__init__()

    def calculate_loss(self, outputs, targets):
        input = outputs["logit"]
        target = targets["labels"].float().cuda()
        probs = torch.sigmoid(input)
        return 1 - (sensitivity(probs, target) + specificity(probs, target))/2
