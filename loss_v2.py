from typing import Dict

import torch
import torch.nn as nn


def masked_priority_bce_loss(
    priority_logits: torch.Tensor,
    y_priority: torch.Tensor,
    valid_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    priority_logits: [B,100]
    y_priority:      [B,100] float
    valid_mask:      [B,100] bool
    """
    logits_v = priority_logits[valid_mask]
    target_v = y_priority[valid_mask]

    if logits_v.numel() == 0:
        return priority_logits.sum() * 0.0

    if pos_weight is None:
        bce = nn.BCEWithLogitsLoss()
    else:
        bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return bce(logits_v, target_v)


def count_mse_loss(
    count_pred: torch.Tensor,
    y_count: torch.Tensor,
) -> torch.Tensor:
    """
    count_pred: [B]
    y_count:    [B]
    """
    return nn.functional.mse_loss(count_pred.float(), y_count.float())


def capacity_ce_loss(
    capacity_logits: torch.Tensor,
    y_capacity: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    capacity_logits: [B,100,4]
    y_capacity:      [B,100]
    """
    B, P, C = capacity_logits.shape
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return ce(capacity_logits.reshape(B * P, C), y_capacity.reshape(B * P))


@torch.no_grad()
def compute_priority_metrics(
    priority_logits: torch.Tensor,
    y_priority: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Dict[str, float]:
    logits_v = priority_logits[valid_mask]
    target_v = y_priority[valid_mask]

    if logits_v.numel() == 0:
        return {"priority_acc": 0.0}

    pred_v = (torch.sigmoid(logits_v) > 0.5).float()
    acc = (pred_v == target_v).float().mean().item()
    return {"priority_acc": acc}


@torch.no_grad()
def compute_count_metrics(
    count_pred: torch.Tensor,
    y_count: torch.Tensor,
) -> Dict[str, float]:
    mae = (count_pred.float() - y_count.float()).abs().mean().item()
    return {"count_mae": mae}


@torch.no_grad()
def compute_capacity_metrics(
    capacity_logits: torch.Tensor,
    y_capacity: torch.Tensor,
    ignore_index: int = -100,
) -> Dict[str, float]:
    pred = capacity_logits.argmax(dim=-1)
    valid = (y_capacity != ignore_index)

    if valid.sum().item() == 0:
        return {"capacity_acc": 0.0}

    acc = (pred[valid] == y_capacity[valid]).float().mean().item()
    return {"capacity_acc": acc}