"""
Data augmentation transforms for training and validation
Extreme augmentation for small datasets
"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_training_transform(image_size=512):
    """
    Extreme data augmentation for training (small dataset optimization)
    
    Args:
        image_size: Target size for images (default: 512)
    
    Returns:
        Albumentations Compose object with heavy augmentation
    """
    return A.Compose([
        # Resize
        A.Resize(image_size, image_size),
        
        # Geometric transformations (aggressive)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.2,
            scale_limit=0.3,
            rotate_limit=180,  # Full rotation
            border_mode=0,
            p=0.7
        ),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            p=0.5
        ),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(distort_limit=0.3, shift_limit=0.3, p=0.3),
        
        # Color augmentation (aggressive)
        A.RandomBrightnessContrast(
            brightness_limit=0.3,
            contrast_limit=0.3,
            p=0.7
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=0.5
        ),
        
        # Noise and blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        
        # Dropout and cutout
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            fill_value=0,
            p=0.3
        ),
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        
        # Sharpen
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        
        # Normalization (ImageNet stats)
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_validation_transform(image_size=512):
    """
    Validation transform (no augmentation, only normalize)
    
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
