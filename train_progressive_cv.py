"""
Progressive Training with K-Fold Cross-Validation
Train lumen segmentation model with progressive resizing and cross-validation for best results
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
import segmentation_models_pytorch as smp
from sklearn.model_selection import KFold
import json
from datetime import datetime

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


def train_single_phase(model, phase_config, phase_index, training_dataset_indices,
                       validation_dataset_indices, all_image_paths, all_mask_paths, 
                       loss_criterion, device, fold_number):
    """
    Train model for a single progressive phase
    
    Args:
        model: U-Net model to train
        phase_config: Dictionary with phase configuration
        phase_index: Index of current phase
        training_dataset_indices: Indices for training data
        validation_dataset_indices: Indices for validation data
        all_image_paths: All image paths
        all_mask_paths: All mask paths
        loss_criterion: Loss function
        device: Device to train on
        fold_number: Current fold number
    
    Returns:
        tuple: (best_dice_this_phase, phase_results_list)
    """
    phase_image_size = phase_config['image_size']
    phase_num_epochs = phase_config['num_epochs']
    phase_learning_rate = phase_config['initial_learning_rate']
    phase_batch_size = phase_config['batch_size']
    phase_description = phase_config['phase_name']
    
    print(f"\n{'-'*80}")
    print(f"{phase_description}")
    print(f"Image Size: {phase_image_size}×{phase_image_size}")
    print(f"Learning Rate: {phase_learning_rate}")
    print(f"Batch Size: {phase_batch_size}")
    print(f"Epochs: {phase_num_epochs}")
    print(f"{'-'*80}")
    
    # Create full datasets with phase-specific transform
    full_training_dataset = LumenDataset(
        all_image_paths,
        all_mask_paths,
        transform=get_training_transform(phase_image_size)
    )
    
    full_validation_dataset = LumenDataset(
        all_image_paths,
        all_mask_paths,
        transform=get_validation_transform(phase_image_size)
    )
    
    # Create subsets for this fold
    training_dataset_subset = Subset(full_training_dataset, training_dataset_indices)
    validation_dataset_subset = Subset(full_validation_dataset, validation_dataset_indices)
    
    # Create data loaders
    training_dataloader = DataLoader(
        training_dataset_subset,
        batch_size=phase_batch_size,
        shuffle=True,
        num_workers=ProgressiveTrainingConfig.NUM_WORKERS,
        pin_memory=ProgressiveTrainingConfig.PIN_MEMORY
    )
    
    validation_dataloader = DataLoader(
        validation_dataset_subset,
        batch_size=phase_batch_size,
        shuffle=False,
        num_workers=ProgressiveTrainingConfig.NUM_WORKERS,
        pin_memory=ProgressiveTrainingConfig.PIN_MEMORY
    )
    
    # Setup optimizer and scheduler
    phase_optimizer = torch.optim.Adam(model.parameters(), lr=phase_learning_rate)
    phase_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        phase_optimizer,
        T_max=phase_num_epochs
    )
    
    best_dice_this_phase = 0.0
    phase_results_list = []
    
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
        
        # Track best dice for this phase
        if validation_dice > best_dice_this_phase:
            best_dice_this_phase = validation_dice
        
        # Record epoch results
        epoch_result = {
            'fold_number': fold_number,
            'phase_index': phase_index + 1,
            'phase_image_size': phase_image_size,
            'epoch_number': current_epoch,
            'training_loss': training_loss,
            'training_dice': training_dice,
            'validation_loss': validation_loss,
            'validation_dice': validation_dice,
            'validation_iou': validation_iou,
        }
        phase_results_list.append(epoch_result)
    
    return best_dice_this_phase, phase_results_list


def train_single_fold(fold_index, training_dataset_indices, validation_dataset_indices,
                     all_image_paths, all_mask_paths, device):
    """
    Train one fold with progressive resizing through all phases
    
    Args:
        fold_index: Index of current fold (0-based)
        training_dataset_indices: Indices for training samples
        validation_dataset_indices: Indices for validation samples
        all_image_paths: All image paths
        all_mask_paths: All mask paths
        device: Device to train on
    
    Returns:
        tuple: (best_validation_dice_score, fold_results_list)
    """
    fold_number = fold_index + 1
    num_total_folds = ProgressiveTrainingConfig.NUM_FOLDS
    
    print(f"\n{'='*80}")
    print(f"FOLD {fold_number}/{num_total_folds}")
    print(f"{'='*80}")
    print(f"Training samples: {len(training_dataset_indices)}")
    print(f"Validation samples: {len(validation_dataset_indices)}")
    
    # Create model for this fold
    model = create_unet_model(
        ProgressiveTrainingConfig.ENCODER_NAME,
        ProgressiveTrainingConfig.ENCODER_PRETRAIN_WEIGHTS,
        device
    )
    
    loss_criterion = smp.losses.DiceLoss(mode='binary')
    best_validation_dice_overall = 0.0
    all_fold_results = []
    
    # Train through progressive phases
    for phase_index, phase_config in enumerate(ProgressiveTrainingConfig.TRAINING_PHASES):
        print(f"\n{'='*80}")
        print(f"PHASE {phase_index+1}/{len(ProgressiveTrainingConfig.TRAINING_PHASES)} - FOLD {fold_number}")
        print(f"{'='*80}")
        
        best_phase_dice, phase_results = train_single_phase(
            model=model,
            phase_config=phase_config,
            phase_index=phase_index,
            training_dataset_indices=training_dataset_indices,
            validation_dataset_indices=validation_dataset_indices,
            all_image_paths=all_image_paths,
            all_mask_paths=all_mask_paths,
            loss_criterion=loss_criterion,
            device=device,
            fold_number=fold_number
        )
        
        all_fold_results.extend(phase_results)
        
        # Update overall best for this fold
        if best_phase_dice > best_validation_dice_overall:
            best_validation_dice_overall = best_phase_dice
            
            # Save best model for this fold
            checkpoint_data = {
                'fold_number': fold_number,
                'phase_index': phase_index,
                'model_state_dict': model.state_dict(),
                'validation_dice': best_validation_dice_overall,
                'encoder_name': ProgressiveTrainingConfig.ENCODER_NAME,
            }
            
            model_filename = f'fold{fold_number}_best_model.pth'
            model_save_path = os.path.join(
                ProgressiveTrainingConfig.MODELS_OUTPUT_DIR,
                model_filename
            )
            torch.save(checkpoint_data, model_save_path)
            print(f"✓ Saved best model for fold {fold_number}! (Dice: {best_validation_dice_overall:.4f})")
    
    print(f"\nFold {fold_number} completed. Best Validation Dice: {best_validation_dice_overall:.4f}")
    
    return best_validation_dice_overall, all_fold_results


def save_cross_validation_results(all_fold_dice_scores, all_detailed_results):
    """
    Save cross-validation results to JSON file
    
    Args:
        all_fold_dice_scores: List of best dice scores for each fold
        all_detailed_results: List of all epoch results across all folds
    
    Returns:
        str: Path to saved results file
    """
    mean_dice_score = np.mean(all_fold_dice_scores)
    std_dice_score = np.std(all_fold_dice_scores)
    
    results_dictionary = {
        'configuration': {
            'num_folds': ProgressiveTrainingConfig.NUM_FOLDS,
            'encoder_name': ProgressiveTrainingConfig.ENCODER_NAME,
            'training_phases': ProgressiveTrainingConfig.TRAINING_PHASES,
            'random_seed': ProgressiveTrainingConfig.RANDOM_SEED,
        },
        'fold_dice_scores': all_fold_dice_scores,
        'mean_dice_score': mean_dice_score,
        'std_dice_score': std_dice_score,
        'best_fold_dice': max(all_fold_dice_scores),
        'worst_fold_dice': min(all_fold_dice_scores),
        'detailed_epoch_results': all_detailed_results,
    }
    
    timestamp_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f'cv_results_{timestamp_string}.json'
    results_filepath = os.path.join(
        ProgressiveTrainingConfig.RESULTS_OUTPUT_DIR,
        results_filename
    )
    
    with open(results_filepath, 'w') as results_file:
        json.dump(results_dictionary, results_file, indent=2)
    
    return results_filepath


def print_final_summary(all_fold_dice_scores, results_filepath):
    """
    Print final cross-validation summary
    
    Args:
        all_fold_dice_scores: List of best dice scores for each fold
        results_filepath: Path to saved results file
    """
    mean_dice_score = np.mean(all_fold_dice_scores)
    std_dice_score = np.std(all_fold_dice_scores)
    best_fold_dice = max(all_fold_dice_scores)
    worst_fold_dice = min(all_fold_dice_scores)
    
    print("\n" + "="*80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("="*80)
    
    for fold_num, dice_score in enumerate(all_fold_dice_scores, 1):
        status_icon = "🏆" if dice_score == best_fold_dice else ("⭐" if dice_score >= mean_dice_score else "")
        print(f"Fold {fold_num}: {dice_score:.4f} ({dice_score*100:.2f}%) {status_icon}")
    
    print(f"\n{'─'*80}")
    print(f"Mean Dice Score:   {mean_dice_score:.4f} ± {std_dice_score:.4f}")
    print(f"Best Fold:         {best_fold_dice:.4f} ({best_fold_dice*100:.2f}%)")
    print(f"Worst Fold:        {worst_fold_dice:.4f} ({worst_fold_dice*100:.2f}%)")
    print(f"{'─'*80}")
    
    if mean_dice_score >= 0.95:
        print("\n🎉 EXCELLENT! Mean Dice ≥ 95% - Publication-ready results!")
    elif mean_dice_score >= 0.90:
        print("\n✓ VERY GOOD! Mean Dice ≥ 90% - Strong performance!")
    elif mean_dice_score >= 0.85:
        print("\n⚠️  GOOD. Mean Dice ≥ 85% - Consider ensemble or more training.")
    else:
        print(f"\n❌ NEEDS IMPROVEMENT. Mean Dice: {mean_dice_score:.1%}")
        print("   Suggestions: Check data quality, try different encoder, or train longer")
    
    print(f"\n✓ Detailed results saved to: {results_filepath}")
    print("="*80)


def main():
    """Main training function with K-fold cross-validation"""
    
    print("="*80)
    print("PROGRESSIVE TRAINING WITH K-FOLD CROSS-VALIDATION")
    print("="*80)
    print(f"Device: {Config.DEVICE}")
    print(f"Encoder: {ProgressiveTrainingConfig.ENCODER_NAME}")
    print(f"Number of Folds: {ProgressiveTrainingConfig.NUM_FOLDS}")
    print(f"Total Phases: {len(ProgressiveTrainingConfig.TRAINING_PHASES)}")
    print(f"Total Epochs per Fold: {ProgressiveTrainingConfig.get_total_epochs()}")
    print()
    
    # Display phase information
    for phase_idx, phase_config in enumerate(ProgressiveTrainingConfig.TRAINING_PHASES, 1):
        print(f"  Phase {phase_idx}: {phase_config['image_size']}×{phase_config['image_size']}, "
              f"{phase_config['num_epochs']} epochs, LR={phase_config['initial_learning_rate']}")
    print()
    
    # Create output directories
    os.makedirs(ProgressiveTrainingConfig.RESULTS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(ProgressiveTrainingConfig.MODELS_OUTPUT_DIR, exist_ok=True)
    
    # Load all dataset paths
    all_image_paths, all_mask_paths = load_dataset_paths(
        Config.IMAGES_DIR,
        Config.MASKS_DIR
    )
    
    total_images_count = len(all_image_paths)
    print(f"Total images found: {total_images_count}")
    print()
    
    # Setup K-Fold cross-validation
    kfold_splitter = KFold(
        n_splits=ProgressiveTrainingConfig.NUM_FOLDS,
        shuffle=True,
        random_state=ProgressiveTrainingConfig.RANDOM_SEED
    )
    
    all_fold_dice_scores = []
    all_detailed_results = []
    
    # Train each fold
    for fold_index, (training_indices, validation_indices) in enumerate(
        kfold_splitter.split(all_image_paths)
    ):
        fold_best_dice, fold_results = train_single_fold(
            fold_index=fold_index,
            training_dataset_indices=training_indices,
            validation_dataset_indices=validation_indices,
            all_image_paths=all_image_paths,
            all_mask_paths=all_mask_paths,
            device=Config.DEVICE
        )
        
        all_fold_dice_scores.append(fold_best_dice)
        all_detailed_results.extend(fold_results)
    
    # Save results
    results_filepath = save_cross_validation_results(
        all_fold_dice_scores,
        all_detailed_results
    )
    
    # Print final summary
    print_final_summary(all_fold_dice_scores, results_filepath)


if __name__ == "__main__":
    main()
