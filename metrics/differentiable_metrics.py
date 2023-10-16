import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any

class FocalLoss(nn.Module):
    def __init__(
            self, 
            alpha: float=.25, 
            gamma: float=2.0, 
            with_logits: bool=False, 
            reduction: str="mean"):
        
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.with_logits = with_logits

    def forward(
            self, 
            input: torch.Tensor, 
            target: torch.Tensor) -> torch.Tensor:
        
        if input.ndim > 2:
            input = input.reshape(-1, input.shape[-1])

        if self.with_logits:
            input = nn.functional.softmax(input, dim=-1)
        prob = input.gather(-1, target.reshape(-1, 1))
        fl = -self.alpha * ((1 - prob)**self.gamma) * torch.log(prob)
        
        if self.reduction == "mean": return fl.mean()
        if self.reduction == "sum": return fl.sum()


class BinaryFocalLoss(nn.Module):
    def __init__(
            self, 
            alpha: float=.25, 
            gamma: float=2.0, 
            with_logits: bool=False,
            epsilon: float=1e-8,
            label_smoothing: Optional[float]=None,
            reduction: str="mean"):
        
        super(BinaryFocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits
        self.epsilon = epsilon
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
            self, input: torch.Tensor, 
            target: torch.Tensor) -> torch.Tensor:
        
        if self.with_logits:
            input = nn.functional.sigmoid(input)

        if self.label_smoothing is not None:
            target = (self.label_smoothing * .5) + (1 - self.label_smoothing) * target

        epsilon = torch.ones_like(input) * self.epsilon

        p = input
        q = 1 - p
        p = torch.max(p, epsilon)
        q = torch.max(q, epsilon)

        pos_loss = -self.alpha * target * ((q)**self.gamma) * torch.log(p)
        neg_loss = -(1 - target) * (p**self.gamma) * torch.log(q)
        fl = pos_loss + neg_loss

        if self.reduction == "mean": return fl.mean()
        if self.reduction == "sum": return fl.sum()


class HybridBinaryEntropyLoss(nn.Module):
    def __init__(
            self, 
            bce_focal_weights: Tuple[float, float]=(0.5, 0.5),
            with_logits: bool=False,
            label_smoothing: Optional[float]=None,
            reduction: str="mean", 
            bce_loss_kwargs: Dict[str, Any]={},
            focal_loss_kwargs: Dict[str, Any]={}):
        
        super(HybridBinaryEntropyLoss, self).__init__()

        assert sum(bce_focal_weights) == 1, "sum of weights in bce_focal_weights must be equal to 1"

        self.bce_focal_weights = bce_focal_weights
        self.with_logits = with_logits
        self.label_smoothing = label_smoothing

        self.focal_loss_kwargs = {
            **dict(alpha=.25, gamma=2.0, epsilon=1e-8,), 
            **focal_loss_kwargs,
        }

        self.bce_loss_kwargs =  {
            **dict(),
            **bce_loss_kwargs,
        }

        if with_logits:
            self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction, **self.bce_loss_kwargs)
        else:
            self.bce_loss = nn.BCELoss(reduction=reduction, **self.bce_loss_kwargs)

        self.focal_loss = BinaryFocalLoss(
            with_logits=with_logits, label_smoothing=None, reduction=reduction, **self.focal_loss_kwargs
        )

    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing is not None:
            target = (self.label_smoothing * .5) + (1 - self.label_smoothing) * target

        bce_w, focal_w = self.bce_focal_weights

        bce = self.bce_loss(input, target)
        fl = self.focal_loss(input, target)

        loss = (bce_w * bce) + (focal_w * fl)
        return loss