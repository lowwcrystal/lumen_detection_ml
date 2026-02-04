"""
Training and validation functions for model training
"""
import torch
from tqdm import tqdm
from .metrics import dice_coefficient, iou_score


def train_single_epoch(model, dataloader, loss_criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: PyTorch model to train
        dataloader: Training data loader
        loss_criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)
    
    Returns:
        tuple: (average_loss, average_dice_score)
    """
    model.train()
    epoch_total_loss = 0.0
    epoch_total_dice = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for batch_images, batch_masks in progress_bar:
        batch_images = batch_images.to(device)
        batch_masks = batch_masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        model_outputs = model(batch_images)
        batch_loss = loss_criterion(model_outputs, batch_masks)
        
        # Backward pass
        batch_loss.backward()
        optimizer.step()
        
        # Calculate metrics
        batch_dice = dice_coefficient(model_outputs, batch_masks)
        epoch_total_loss += batch_loss.item()
        epoch_total_dice += batch_dice
        
        progress_bar.set_postfix({
            'loss': f'{batch_loss.item():.4f}',
            'dice': f'{batch_dice:.4f}'
        })
    
    average_epoch_loss = epoch_total_loss / num_batches
    average_epoch_dice = epoch_total_dice / num_batches
    
    return average_epoch_loss, average_epoch_dice


def validate_single_epoch(model, dataloader, loss_criterion, device):
    """
    Validate model for one epoch
    
    Args:
        model: PyTorch model to validate
        dataloader: Validation data loader
        loss_criterion: Loss function
        device: Device to run on (cuda/cpu)
    
    Returns:
        tuple: (average_loss, average_dice_score, average_iou_score)
    """
    model.eval()
    epoch_total_loss = 0.0
    epoch_total_dice = 0.0
    epoch_total_iou = 0.0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        
        for batch_images, batch_masks in progress_bar:
            batch_images = batch_images.to(device)
            batch_masks = batch_masks.to(device)
            
            # Forward pass
            model_outputs = model(batch_images)
            batch_loss = loss_criterion(model_outputs, batch_masks)
            
            # Calculate metrics
            batch_dice = dice_coefficient(model_outputs, batch_masks)
            batch_iou = iou_score(model_outputs, batch_masks)
            
            epoch_total_loss += batch_loss.item()
            epoch_total_dice += batch_dice
            epoch_total_iou += batch_iou
            
            progress_bar.set_postfix({
                'loss': f'{batch_loss.item():.4f}',
                'dice': f'{batch_dice:.4f}'
            })
    
    average_epoch_loss = epoch_total_loss / num_batches
    average_epoch_dice = epoch_total_dice / num_batches
    average_epoch_iou = epoch_total_iou / num_batches
    
    return average_epoch_loss, average_epoch_dice, average_epoch_iou
