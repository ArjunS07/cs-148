# Adversarial Dataset Generation via Frequency Spectrum Modification

This project modifies the frequency spectrum of custom images to maximize their dissimilarity from ground truth images while preserving human recognizability. The goal is to create adversarial examples that fool CNNs but remain recognizable to humans.

## Approach

1. **Frequency Segmentation**: Divide the frequency spectrum into log-spaced radial buckets
2. **Feature Extraction**: Use MobileNetV2 to extract deep features for distribution comparison
3. **Dissimilarity Metric**: Maximize MMD (Maximum Mean Discrepancy) between modified and ground truth feature distributions
4. **Recognizability Constraints**: Enforce SSIM, low-frequency correlation, and template matching thresholds
5. **Optimization**: Gradient ascent with central finite differences and backtracking

## Project Structure

```
adversarial_mnist/
├── notebook_content.py     # Main notebook content (copy to .ipynb cells)
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/
│   ├── __init__.py
│   ├── config.py          # Hyperparameters and paths
│   ├── data_loader.py     # Dataset loading utilities
│   ├── feature_extractor.py # MobileNetV2 feature extraction
│   ├── frequency.py       # FFT and frequency manipulation
│   ├── metrics.py         # MMD, SSIM, recognizability checks
│   ├── optimizer.py       # Gradient estimation and optimization loop
│   └── visualization.py   # Plotting and image saving
└── outputs/               # Created at runtime
    ├── modified_images/   # Final modified images
    ├── iterations/        # Per-iteration snapshots
    ├── figures/           # Visualization figures
    ├── weights.npy        # Final optimized weights
    └── optimization_history.json
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your data:
   - Ground truth dataset: Directory with subdirectories `0/`, `1/`, etc. containing images
   - Custom dataset: Same structure, images you want to modify

3. Create a Jupyter notebook and copy content from `notebook_content.py`
   - Each `# CELL:` marker indicates a new cell
   - `# CELL: Markdown` cells should be converted to markdown
   - `# CELL: Code` cells are Python code

4. Edit the paths in the configuration cell:
```python
GROUND_TRUTH_DIR = "./ground_truth"  # Your ground truth path
CUSTOM_DIR = "./custom"              # Your custom dataset path
```

## Configuration

Key hyperparameters in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMAGE_SIZE` | 256 | Expected image dimensions |
| `BUCKET_BOUNDARIES` | [0,2,4,8,16,32,64,128,181] | Radial frequency bucket boundaries |
| `NUM_ITERATIONS` | 15 | Optimization iterations |
| `LR_INITIAL` | 0.3 | Initial learning rate |
| `LR_FINAL` | 0.03 | Final learning rate |
| `WEIGHT_MIN/MAX` | 0.1 / 3.0 | Weight bounds |
| `SSIM_THRESHOLD` | 0.6 | Minimum SSIM for recognizability |
| `LOW_FREQ_CORRELATION_THRESHOLD` | 0.9 | Minimum low-frequency correlation |
| `TEMPLATE_MATCH_THRESHOLD` | 0.9 | Minimum template matching accuracy |

## Output

After running the notebook:

- **Modified images**: `outputs/modified_images/{label}/*.png`
- **Weights**: `outputs/weights.npy` (numpy array) and `outputs/weights.json`
- **Figures**: Various visualizations in `outputs/figures/`
- **History**: `outputs/optimization_history.json` with metrics per iteration

## How It Works

### Frequency Buckets
The 2D FFT of each image is divided into concentric rings based on radial distance from the DC component. Lower buckets contain low frequencies (overall shape, gradients), higher buckets contain high frequencies (edges, textures).

### Optimization
For each label, we learn a weight vector `w = [w_0, w_1, ..., w_k]` where each `w_i` scales the magnitude of frequencies in bucket `i`. We maximize MMD while enforcing recognizability constraints:

```
maximize MMD(modified_features, ground_truth_features)
subject to:
    SSIM(original, modified) >= 0.6
    low_freq_correlation(original, modified) >= 0.9
    template_match_accuracy(modified) >= 0.9
```

### Gradient Estimation
Since the objective involves eigendecomposition and is non-differentiable, we use central finite differences:

```
∂f/∂w_i ≈ (f(w + ε*e_i) - f(w - ε*e_i)) / (2ε)
```

## References

- Wang et al., "High Frequency Component Helps Explain the Generalization of CNNs" (2020)
- Gretton et al., "A Kernel Two-Sample Test" (2012) - MMD
