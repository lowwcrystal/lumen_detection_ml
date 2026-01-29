# Lumen Segmentation for Fibrosis Quantification

This project trains a U-Net model to detect lumens in Sirius Red stained images, allowing for accurate fibrosis quantification by subtracting lumen red pixels from total red measurements.

## Project Structure

```
cell segmentation/
├── images/                          # Original annotated images + JSON files
├── output_masks/                    # Generated masks from LabelMe annotations
│   ├── JPEGImages/                 # Training images
│   └── SegmentationClass/          # Training masks
├── models/                          # Saved model checkpoints (created during training)
├── predictions/                     # Prediction outputs (created during inference)
├── train_lumen_model.py            # Training script
├── predict_lumen.py                # Inference script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you have an NVIDIA GPU, install PyTorch with CUDA support first:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Then install the rest:
```bash
pip install segmentation-models-pytorch albumentations opencv-python Pillow numpy scikit-learn tqdm
```

## Usage

### Training the Model

To train the lumen segmentation model on your annotated data:

```bash
python3 train_lumen_model.py
```

**What happens:**
- Loads 28 annotated images and masks
- Splits into 80% training, 20% validation
- Applies data augmentation to increase dataset size
- Trains U-Net with ResNet34 encoder for 100 epochs
- Saves the best model to `models/best_lumen_model.pth`
- Displays training progress with loss, Dice score, and IoU metrics

**Training time:** 
- CPU: ~2-4 hours
- GPU: ~20-40 minutes

### Making Predictions on New Images

Once trained, use the model to predict lumen masks on new images:

#### Single image:
```bash
python3 predict_lumen.py path/to/image.jpg --output predictions --visualize --save-masks
```

#### Batch processing (entire folder):
```bash
python3 predict_lumen.py path/to/images/folder --output predictions --visualize --save-masks
```

**Options:**
- `--output`: Directory to save results (default: `predictions`)
- `--visualize`: Save visualization images with lumen overlay
- `--save-masks`: Save binary mask images
- `--model`: Path to trained model (default: `models/best_lumen_model.pth`)

**Outputs:**
- `results.csv`: CSV file with red pixel calculations for each image
  - `total_red_pixels`: All red pixels in image
  - `lumen_red_pixels`: Red pixels inside detected lumens
  - `fibrosis_red_pixels`: Corrected red pixels (total - lumen)
  - `fibrosis_red_percentage`: Accurate fibrosis percentage
  - `correction_percentage`: How much lumen correction was applied
- `visualizations/`: Images with red overlay showing detected lumens
- `masks/`: Binary masks (white = lumen, black = background)

## How It Works

### Training Pipeline:
1. **Data Loading:** Loads images and corresponding lumen masks
2. **Augmentation:** Applies rotation, flipping, brightness/contrast changes, elastic deformation
3. **Model:** U-Net architecture with pre-trained ResNet34 encoder
4. **Loss:** Dice Loss (optimized for segmentation overlap)
5. **Validation:** Monitors Dice score and IoU to save best model

### Prediction Pipeline:
1. **Segmentation:** Model predicts lumen regions in new image
2. **Red Detection:** Detects all red pixels using HSV color space
3. **Calculation:** 
   - Total red = all red pixels (inflated by lumens)
   - Lumen red = red pixels inside predicted lumens
   - **Fibrosis red = Total red - Lumen red** (accurate measurement)
4. **Export:** Saves results to CSV for analysis

## Customization

### Adjust Training Parameters

Edit `train_lumen_model.py`:

```python
class Config:
    LEARNING_RATE = 0.0001    # Lower = slower but more stable
    BATCH_SIZE = 4            # Increase if you have more GPU memory
    NUM_EPOCHS = 100          # More epochs = longer training
    IMAGE_SIZE = 512          # Higher = more detail but slower
    ENCODER = "resnet34"      # Try "resnet50" or "efficientnet-b0"
```

### Adjust Red Color Detection

Edit `predict_lumen.py` if your Sirius Red staining has different color characteristics:

```python
# Adjust HSV ranges for red detection
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
```

### Prediction Threshold

Adjust sensitivity in `predict_lumen.py`:

```python
class Config:
    THRESHOLD = 0.5  # Lower = more sensitive (more lumens detected)
                     # Higher = more specific (only confident lumens)
```

## Troubleshooting

### Training is too slow
- Reduce `IMAGE_SIZE` to 256 or 384
- Reduce `BATCH_SIZE` to 2
- Use a lighter encoder like `mobilenet_v2`

### Model overfits (train accuracy high, validation low)
- Add more augmentation
- Use dropout or early stopping
- Reduce model complexity (use `mobilenet_v2`)

### Predictions are inaccurate
- Train for more epochs
- Check if validation Dice score is >0.80
- Adjust `THRESHOLD` in prediction script
- Add more training images with diverse examples

### Out of memory errors
- Reduce `BATCH_SIZE` to 2 or 1
- Reduce `IMAGE_SIZE` to 256
- Close other applications

## Results Interpretation

**Good model performance:**
- Validation Dice score: >0.85
- Validation IoU: >0.75
- Correction percentage: 5-20% (typical lumen contribution)

**In your CSV results:**
- `correction_percentage`: Shows how much lumens inflated your original measurements
- `fibrosis_red_percentage`: Use this as your accurate fibrosis measurement
- Compare to your original total red percentage to see the correction impact

## Citation

If you use this for publications, consider citing the segmentation models library:
- segmentation_models_pytorch: https://github.com/qubvel/segmentation_models.pytorch

## Contact

For issues or questions about this pipeline, refer to the original conversation or documentation.
