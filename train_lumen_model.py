import os
from huggingface_hub import file_download
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config import Config
from dataset import LumenDataset
from utils import (
    get_training_transform,
    get_validation_transform,
    train_single_epoch,
    validate_single_epoch
)


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
    train_images, validation_images, train_masks, validation_masks = train_test_split(
        image_paths, mask_paths, test_size=Config.VALIDATION_SPLIT, random_state=32
    )
    
    
    # Create datasets
    train_dataset = LumenDataset(
        train_images, 
        train_masks, 
        transform=get_training_transform(Config.IMAGE_SIZE)
    )
    validation_dataset = LumenDataset(
        validation_images, 
        validation_masks, 
        transform=get_validation_transform(Config.IMAGE_SIZE)
    )

    # Create data loaders
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=True, 
        num_workers=0, 
        pin_memory=False
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=Config.BATCH_SIZE,
        shuffle=False, 
        num_workers=0, 
        pin_memory=False
    )
    
    # Create model
    print("Creating U-Net model")
    model = smp.DeepLabV3Plus(
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
        train_loss, train_dice = train_single_epoch(
            model, train_loader, criterion, optimizer, Config.DEVICE
        )
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        
        # Validate
        validation_loss, validation_dice, validation_iou = validate_single_epoch(
            model, validation_loader, criterion, Config.DEVICE
        )
        print(f"Validation Loss: {validation_loss:.4f}, Validation Dice: {validation_dice:.4f}, Validation IoU: {validation_iou:.4f}")
        
        # Update learning rate
        scheduler.step(validation_loss)
        
        # Save best model
        if validation_dice > best_dice:
            best_dice = validation_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'validation_dice': validation_dice,
                'validation_iou': validation_iou,
            }, Config.MODEL_SAVE_PATH)
            print(f"✓ Saved best model! (Dice: {validation_dice:.4f})")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best validation Dice score: {best_dice:.4f}")
    print(f"Model saved to: {Config.MODEL_SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
