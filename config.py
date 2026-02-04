import torch


class Config:
    # paths
    IMAGES_DIR = "output_masks/JPEGImages"
    MASKS_DIR = "output_masks/SegmentationClass"
    MODEL_SAVE_PATH = "models/best_lumen_model.path"
    
    # training parameters
    ENCODER = "timm-resnest101e"
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = 2
    LEARNING_RATE = 0.001
    BATCH_SIZE = 8
    NUM_EPOCHS = 200
    VALIDATION_SPLIT = 0.2
    
    # image parameters
    IMAGE_SIZE = 512
    
    # device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
