# utils/metrics.py
import torch
from monai.metrics import DiceMetric, compute_iou

def pixel_accuracy(y_pred, y_true, threshold=0.5):
    y_pred = (torch.sigmoid(y_pred) > threshold)
    correct = torch.sum(y_pred == y_true)
    return (correct.float() / y_true.numel()).item()

def iou_score(y_pred, y_true, threshold=0.5):
    y_pred = (torch.sigmoid(y_pred) > threshold)
    return compute_iou(y_pred, y_true).nanmean().item()

def dice_coefficient(y_pred, y_true, threshold=0.5):
    y_pred = (torch.sigmoid(y_pred) > threshold)
    dice_metric = DiceMetric()
    return dice_metric(y_pred, y_true).nanmean().item()