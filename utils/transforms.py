"""
Data augmentation transforms for training and validation
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform(image_size=512):
    """
    Get training data augmentation pipeline
    
    Args:
        image_size: Target size for images (default: 512)
    
    Returns:
        Albumentations Compose object with training transforms
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform(image_size=512):
    """
    Get validation data transform pipeline (no augmentation)
    
    Args:
        image_size: Target size for images (default: 512)
    
    Returns:
        Albumentations Compose object with validation transforms
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
