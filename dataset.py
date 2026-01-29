"""
Dataset class for lumen segmentation
"""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class LumenDataset(Dataset):
    
    def __init__(self, image_paths, mask_paths, transform=None):
        
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, index):
        # Load an image
        image = Image.open(self.image_paths[index]).convert("RGB")
        image = np.array(image)
        
        # Load the corresponding mask
        mask = Image.open(self.mask_paths[index])
        mask = np.array(mask)
        
        # Convert mask to binary where 0 = background, 1 = lumen
        mask = (mask > 0).astype(np.float32)
        
        # Apply augmentations to the image and mask
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.unsqueeze(0) 
