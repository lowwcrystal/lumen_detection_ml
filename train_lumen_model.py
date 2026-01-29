import os
from huggingface_hub import file_download
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import Config
from dataset import LumenDataset
from utils import get_train_transform, get_val_transform, dice_coefficient, iou_score


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train model for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        dice = dice_coefficient(outputs, masks)
        running_loss += loss.item()
        running_dice += dice
        
        pbar.set_postfix({'loss': loss.item(), 'dice': dice})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    return epoch_loss, epoch_dice


def validate_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice, 'iou': iou})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    return epoch_loss, epoch_dice, epoch_iou


def main():
    print("LUMEN SEGMENTATION MODEL TRAINING")
    print(f"Image size: {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Epochs: {Config.NUM_EPOCHS}")
    print()
    
    # Get all image and mask paths
    image_files = sorted([file for file in os.listdir(Config.IMAGES_DIR) if file.endswith('.jpg')])
    image_paths = [os.path.join(Config.IMAGES_DIR, file) for file in image_files]
    
    # Get corresponding mask paths 
    mask_paths = []
    for img_file in image_files:
        mask_file = img_file.replace('.jpg', '.png')
        mask_path = os.path.join(Config.MASKS_DIR, mask_file)
        mask_paths.append(mask_path)
    
    
    # Split data into train and validation
    train_images, val_images, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=Config.VALIDATION_SPLIT, random_state=32
    )
    
    
    # Create datasets
    train_dataset = LumenDataset(
        train_images, 
        train_masks, 
        transform=get_train_transform(Config.IMAGE_SIZE)
    )
    val_dataset = LumenDataset(
        val_images, 
        val_masks, 
        transform=get_val_transform(Config.IMAGE_SIZE)
    )
    

    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    
    # Create model
    print("Creating U-Net model")
    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=Config.ENCODER_WEIGHTS,
        in_channels=3,
        classes=1,  
    )
    model = model.to(Config.DEVICE)
    
    # Loss function and optimizer to check metrics during training
    criterion = smp.losses.DiceLoss(mode='binary')
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,      
        T_mult=2,
        eta_min=1e-6
    )
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Training loop
    best_dice = 0.0
    print("\nStarting training...")
    print("=" * 60)
    
    for epoch in range(Config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.NUM_EPOCHS}")
        print("-" * 60)
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validate
        val_loss, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, Config.DEVICE
        )
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
            }, Config.MODEL_SAVE_PATH)
            print(f"✓ Saved best model! (Dice: {val_dice:.4f})")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
