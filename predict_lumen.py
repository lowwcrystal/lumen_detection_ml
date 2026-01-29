"""
Lumen Segmentation Inference Script
Use trained model to predict lumen masks on new images and calculate red pixels within lumens
"""

import os
import argparse
import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm

# Import local modules
from config import Config

# ============================================================================
# PREDICTION CONFIGURATION
# ============================================================================
THRESHOLD = 0.5  # Probability threshold for binary segmentation

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def get_inference_transform():
    """Transform for inference (no augmentation)"""
    return A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def load_model(model_path):
    """Load trained model"""
    print(f"Loading model from {model_path}...")
    model = smp.Unet(
        encoder_name=Config.ENCODER,
        encoder_weights=None,  # We'll load our trained weights
        in_channels=3,
        classes=1,
    )
    
    checkpoint = torch.load(model_path, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(Config.DEVICE)
    model.eval()
    
    print(f"Model loaded! (Validation Dice: {checkpoint.get('val_dice', 'N/A')})")
    return model

def predict_image(model, image_path, transform):
    """Predict lumen mask for a single image"""
    # Load and preprocess image
    original_image = Image.open(image_path).convert("RGB")
    original_size = original_image.size  # (width, height)
    
    image = np.array(original_image)
    
    # Apply transforms
    augmented = transform(image=image)
    input_tensor = augmented['image'].unsqueeze(0).to(Config.DEVICE)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.sigmoid(output).cpu().numpy()[0, 0]
    
    # Threshold and resize back to original size
    pred_mask = (pred_mask > THRESHOLD).astype(np.uint8) * 255
    pred_mask = cv2.resize(pred_mask, original_size, interpolation=cv2.INTER_NEAREST)
    
    return pred_mask, np.array(original_image)

def calculate_red_pixels(image, mask):
    """
    Calculate red pixels within lumen regions only
    
    Args:
        image: RGB image (numpy array)
        mask: Binary mask (255 = lumen, 0 = background)
    
    Returns:
        dict with lumen red pixel counts
    """
    # Convert to HSV for better red detection
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Define red color range (adjust these values based on your staining)
    # Red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red pixels
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Calculate red pixels in lumen regions ONLY
    lumen_mask_binary = (mask > 0).astype(np.uint8)
    lumen_red_pixels = np.sum((red_mask > 0) & (lumen_mask_binary > 0))
    
    # Calculate lumen area
    total_pixels = image.shape[0] * image.shape[1]
    lumen_area = np.sum(lumen_mask_binary > 0)
    
    results = {
        'lumen_red_pixels': int(lumen_red_pixels),
        'lumen_area_pixels': int(lumen_area),
        'total_pixels': int(total_pixels),
        'lumen_percentage': (lumen_area / total_pixels) * 100,
        'lumen_red_percentage': (lumen_red_pixels / total_pixels) * 100,
    }
    
    return results

def create_visualization(image, mask, output_path):
    """Create visualization with mask overlay"""
    # Create colored overlay (red for lumens)
    overlay = image.copy()
    overlay[mask > 0] = [255, 0, 0]  # Red overlay for lumens
    
    # Blend original and overlay
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Save visualization
    Image.fromarray(blended).save(output_path)

# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Predict lumen masks on new images')
    parser.add_argument('input', help='Input image file or directory')
    parser.add_argument('--output', default='predictions', help='Output directory for results')
    parser.add_argument('--model', default=Config.MODEL_SAVE_PATH, help='Path to trained model')
    parser.add_argument('--visualize', action='store_true', help='Save visualization images')
    parser.add_argument('--save-masks', action='store_true', help='Save binary masks')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LUMEN RED PIXEL DETECTION")
    print("=" * 60)
    print(f"Device: {Config.DEVICE}")
    print(f"Model: {args.model}")
    print("Output: Red pixels within detected lumens only")
    print()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    if args.visualize:
        os.makedirs(os.path.join(args.output, 'visualizations'), exist_ok=True)
    if args.save_masks:
        os.makedirs(os.path.join(args.output, 'masks'), exist_ok=True)
    
    # Load model
    model = load_model(args.model)
    transform = get_inference_transform()
    
    # Get list of images to process
    if os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        image_paths = [
            os.path.join(args.input, f) 
            for f in os.listdir(args.input) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff'))
        ]
    
    print(f"Processing {len(image_paths)} images...\n")
    
    # Process each image
    results_summary = []
    
    for image_path in tqdm(image_paths, desc="Predicting"):
        filename = os.path.basename(image_path)
        base_name = os.path.splitext(filename)[0]
        
        # Predict mask
        pred_mask, original_image = predict_image(model, image_path, transform)
        
        # Calculate red pixels
        results = calculate_red_pixels(original_image, pred_mask)
        results['filename'] = filename
        results_summary.append(results)
        
        # Save mask
        if args.save_masks:
            mask_path = os.path.join(args.output, 'masks', f"{base_name}_mask.png")
            Image.fromarray(pred_mask).save(mask_path)
        
        # Save visualization
        if args.visualize:
            vis_path = os.path.join(args.output, 'visualizations', f"{base_name}_vis.jpg")
            create_visualization(original_image, pred_mask, vis_path)
    
    # Save results to CSV
    import csv
    csv_path = os.path.join(args.output, 'results.csv')
    with open(csv_path, 'w', newline='') as f:
        if results_summary:
            writer = csv.DictWriter(f, fieldnames=results_summary[0].keys())
            writer.writeheader()
            writer.writerows(results_summary)
    
    print(f"\n✓ Results saved to {csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_lumen_area = np.mean([r['lumen_percentage'] for r in results_summary])
    avg_lumen_red = np.mean([r['lumen_red_percentage'] for r in results_summary])
    total_lumen_red = sum([r['lumen_red_pixels'] for r in results_summary])
    print(f"Total images processed: {len(results_summary)}")
    print(f"Average lumen area: {avg_lumen_area:.2f}%")
    print(f"Average red in lumens: {avg_lumen_red:.2f}%")
    print(f"Total lumen red pixels: {total_lumen_red:,}")
    print("=" * 60)

if __name__ == "__main__":
    main()
