"""
Evaluation metrics for segmentation models
"""
import torch


def dice_coefficient(predicted, target, smooth=1e-6):
    """
    Calculate Dice coefficient (F1 score for segmentation)
    
    Args:
        predicted: Predicted segmentation logits (before sigmoid)
        target: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Dice coefficient as float
    """
    predicted = torch.sigmoid(predicted)
    predicted = (predicted > 0.5).float()
    
    intersection = (predicted * target).sum()
    dice = (2. * intersection + smooth) / (predicted.sum() + target.sum() + smooth)
    return dice.item()


def iou_score(predicted, target, smooth=1e-6):
    
    predicted = torch.sigmoid(predicted)
    predicted = (predicted > 0.5).float()
    
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def pixel_accuracy(predicted, target):
    predicted = torch.sigmoid(predicted)
    predicted = (predicted > 0.5).float()
    
    correct = (predicted == target).float().sum()
    total = target.numel()
    return (correct / total).item()
