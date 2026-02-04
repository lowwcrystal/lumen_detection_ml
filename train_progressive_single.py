"""
Progressive Training - Single Model (No Cross-Validation)
Train a single model with progressive image resizing for faster experimentation
"""

import os
import torch
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

from config import Config
from progressive_config import ProgressiveTrainingConfig
from dataset import LumenDataset
from utils import (
    get_training_transform,
    get_validation_transform,
    train_single_epoch,
    validate_single_epoch
)


def create_unet_model(encoder_name, encoder_weights, device):
    """
    Create and initialize U-Net model
    
    Args:
        encoder_name: Name of encoder backbone
        encoder_weights: Pre-trained weights ('imagenet' or None)
        device: Device to place model on
    
    Returns:
        model: Initialized U-Net model
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=ProgressiveTrainingConfig.INPUT_CHANNELS,
        classes=ProgressiveTrainingConfig.OUTPUT_CLASSES,
    )
    model = model.to(device)
    return model


def load_dataset_paths(images_directory, masks_directory):
    """
    Load all image and mask file paths
    
    Args:
        images_directory: Directory containing images
        masks_directory: Directory containing masks
    
    Returns:
        tuple: (image_paths_list, mask_paths_list)
    """
    image_files_list = sorted([
        filename for filename in os.listdir(images_directory)
        if filename.endswith('.jpg')
    ])
    
    image_paths_list = [
        os.path.join(images_directory, filename)
        for filename in image_files_list
    ]
    
    mask_paths_list = []
    for image_filename in image_files_list:
        mask_filename = image_filename.replace('.jpg', '.png')
        mask_file_path = os.path.join(masks_directory, mask_filename)
        mask_paths_list.append(mask_file_path)
    
    return image_paths_list, mask_paths_list


def train_progressive_phases(model, training_image_paths, validation_image_paths,
                            training_mask_paths, validation_mask_paths, device):
    """
    Train model through progressive phases
    
    Args:
        model: U-Net model to train
        training_image_paths: List of training image paths
        validation_image_paths: List of validation image paths
        training_mask_paths: List of training mask paths
        validation_mask_paths: List of validation mask paths
        device: Device to train on
    
    Returns:
        float: Best validation Dice score achieved
    """
    loss_criterion = smp.losses.DiceLoss(mode='binary')
    best_validation_dice = 0.0
    
    for phase_index, phase_config in enumerate(ProgressiveTrainingConfig.TRAINING_PHASES):
        phase_image_size = phase_config['image_size']
        phase_num_epochs = phase_config['num_epochs']
        phase_learning_rate = phase_config['initial_learning_rate']
        phase_batch_size = phase_config['batch_size']
        phase_description = phase_config['phase_name']
        
        print("="*80)
        print(f"PHASE {phase_index+1}/{len(ProgressiveTrainingConfig.TRAINING_PHASES)}")
        print(f"{phase_description}")
        print(f"Image Size: {phase_image_size}×{phase_image_size}")
        print(f"Learning Rate: {phase_learning_rate}")
        print(f"Batch Size: {phase_batch_size}")
        print(f"Epochs: {phase_num_epochs}")
        print("="*80)
        
        # Create datasets for this phase
        training_dataset = LumenDataset(
            training_image_paths,
            training_mask_paths,
            transform=get_training_transform(phase_image_size)
        )
        
        validation_dataset = LumenDataset(
            validation_image_paths,
            validation_mask_paths,
            transform=get_validation_transform(phase_image_size)
        )
        
        # Create data loaders
        training_dataloader = DataLoader(
            training_dataset,
            batch_size=phase_batch_size,
            shuffle=True,
            num_workers=ProgressiveTrainingConfig.NUM_WORKERS,
            pin_memory=ProgressiveTrainingConfig.PIN_MEMORY
        )
        
        validation_dataloader = DataLoader(
            validation_dataset,
            batch_size=phase_batch_size,
            shuffle=False,
            num_workers=ProgressiveTrainingConfig.NUM_WORKERS,
            pin_memory=ProgressiveTrainingConfig.PIN_MEMORY
        )
        
        # Setup optimizer and scheduler for this phase
        phase_optimizer = torch.optim.Adam(model.parameters(), lr=phase_learning_rate)
        phase_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            phase_optimizer,
            T_max=phase_num_epochs
        )
        
        # Train for this phase
        for epoch_number in range(phase_num_epochs):
            current_epoch = epoch_number + 1
            print(f"\nEpoch {current_epoch}/{phase_num_epochs}")
            
            # Training
            training_loss, training_dice = train_single_epoch(
                model, training_dataloader, loss_criterion, phase_optimizer, device
            )
            
            # Validation
            validation_loss, validation_dice, validation_iou = validate_single_epoch(
                model, validation_dataloader, loss_criterion, device
            )
            
            print(f"Training   - Loss: {training_loss:.4f}, Dice: {training_dice:.4f}")
            print(f"Validation - Loss: {validation_loss:.4f}, Dice: {validation_dice:.4f}, IoU: {validation_iou:.4f}")
            
            # Update learning rate
            phase_scheduler.step()
            
            # Save best model
            if validation_dice > best_validation_dice:
                best_validation_dice = validation_dice
                
                checkpoint_data = {
                    'phase_index': phase_index,
                    'epoch_number': epoch_number,
                    'model_state_dict': model.state_dict(),
                    'validation_dice': validation_dice,
                    'validation_iou': validation_iou,
                    'encoder_name': ProgressiveTrainingConfig.ENCODER_NAME,
                }
                
                model_save_path = os.path.join(
                    ProgressiveTrainingConfig.MODELS_OUTPUT_DIR,
                    'progressive_best_single_model.pth'
                )
                torch.save(checkpoint_data, model_save_path)
                print(f"✓ Saved best model! (Dice: {validation_dice:.4f})")
    
    return best_validation_dice


def main():
    """Main training function for progressive single model"""
    
    print("="*80)
    print("PROGRESSIVE TRAINING - SINGLE MODEL")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Encoder: {ProgressiveTrainingConfig.ENCODER_NAME}")
    print(f"Total Phases: {len(ProgressiveTrainingConfig.TRAINING_PHASES)}")
    print(f"Total Epochs: {ProgressiveTrainingConfig.get_total_epochs()}")
    print()
    
    # Display phase information
    for phase_index, phase_config in enumerate(ProgressiveTrainingConfig.TRAINING_PHASES, 1):
        print(f"  Phase {phase_index}: {phase_config['image_size']}×{phase_config['image_size']}, "
              f"{phase_config['num_epochs']} epochs, LR={phase_config['initial_learning_rate']}")
    print()
    
    # Create output directories
    os.makedirs(ProgressiveTrainingConfig.MODELS_OUTPUT_DIR, exist_ok=True)
    
    # Load dataset paths
    all_image_paths, all_mask_paths = load_dataset_paths(
        Config.IMAGES_DIR,
        Config.MASKS_DIR
    )
    
    total_images_count = len(all_image_paths)
    print(f"Total images found: {total_images_count}")
    
    # Split into training and validation
    (training_image_paths, validation_image_paths,
     training_mask_paths, validation_mask_paths) = train_test_split(
        all_image_paths,
        all_mask_paths,
        test_size=ProgressiveTrainingConfig.VALIDATION_SPLIT,
        random_state=ProgressiveTrainingConfig.RANDOM_SEED
    )
    
    training_images_count = len(training_image_paths)
    validation_images_count = len(validation_image_paths)
    
    print(f"Training images: {training_images_count}")
    print(f"Validation images: {validation_images_count}\n")
    
    # Create model
    print("Creating U-Net model...")
    model = create_unet_model(
        ProgressiveTrainingConfig.ENCODER_NAME,
        ProgressiveTrainingConfig.ENCODER_PRETRAIN_WEIGHTS,
        Config.DEVICE
    )
    print("Model created successfully!\n")
    
    # Train through progressive phases
    final_best_dice = train_progressive_phases(
        model,
        training_image_paths,
        validation_image_paths,
        training_mask_paths,
        validation_mask_paths,
        Config.DEVICE
    )
    
    # Final results
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print(f"Best Validation Dice Score: {final_best_dice:.4f} ({final_best_dice*100:.2f}%)")
    
    if final_best_dice >= 0.95:
        print("🎉 EXCELLENT! Dice ≥ 95%")
    elif final_best_dice >= 0.90:
        print("✓ GOOD! Dice ≥ 90%")
    elif final_best_dice >= 0.85:
        print("⚠️  ACCEPTABLE. Dice ≥ 85%. Consider more training.")
    else:
        print(f"❌ LOW. Dice: {final_best_dice:.1%}. Check data quality or try longer training.")
    
    print(f"Model saved to: {ProgressiveTrainingConfig.MODELS_OUTPUT_DIR}/progressive_best_single_model.pth")
    print("="*80)


if __name__ == "__main__":
    main()