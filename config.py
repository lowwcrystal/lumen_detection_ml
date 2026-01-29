import torch


class Config:
    # Paths
    IMAGES_DIR = "output_masks/JPEGImages"
    MASKS_DIR = "output_masks/SegmentationClass"
    MODEL_SAVE_PATH = "models/best_lumen_model.path"
    
    # Training parameters
    ENCODER = "efficientnet-b4"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = 2
    LEARNING_RATE = 0.0001
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    VALIDATION_SPLIT = 0.2
    
    # Image parameters
    IMAGE_SIZE = 512
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
