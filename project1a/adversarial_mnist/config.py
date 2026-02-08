"""
Configuration for adversarial dataset generation.

All hyperparameters and paths are defined here for easy modification.
"""

from pathlib import Path

# =============================================================================
# Paths (set these before running)
# =============================================================================
GROUND_TRUTH_DIR = Path("./ground_truth_imgs")  # Path to ground truth dataset
CUSTOM_DIR = Path("./custom_imgs")        # Path to custom dataset to modify
OUTPUT_DIR = Path("./outputs")

# =============================================================================
# Image Configuration
# =============================================================================
IMAGE_SIZE = 128  # Expected image size (assumes square images)

# =============================================================================
# Frequency Buckets (radial, log-spaced)
# =============================================================================
# For 256x256 images, max frequency is sqrt(2) * 128 ≈ 181
# Log-spaced boundaries covering the full range
BUCKET_BOUNDARIES = [0, 2, 4, 8, 16, 32, 64, 90]

# =============================================================================
# Optimization
# =============================================================================
NUM_ITERATIONS = 85
LR_INITIAL = 1.0
LR_FINAL = 0.01
EPSILON = 0.05  # For finite differences
WEIGHT_MIN = 0.1
WEIGHT_MAX = 3.0

# =============================================================================
# Recognizability Thresholds
# =============================================================================
SSIM_THRESHOLD = 0.6
LOW_FREQ_CORRELATION_THRESHOLD = 0.9
TEMPLATE_MATCH_THRESHOLD = 0.9
LOW_FREQ_CUTOFF_RADIUS = 4  # Scaled for 128x128 (was 4 for 28x28)

# =============================================================================
# Feature Extraction
# =============================================================================
FEATURE_INPUT_SIZE = 224  # MobileNetV2 standard input size
DEVICE = "mps"  # "mps" for Apple Silicon, "cuda" for NVIDIA, "cpu" for fallback

# =============================================================================
# Visualization
# =============================================================================
SAVE_ITERATION_IMAGES = True
FIGURE_DPI = 300


def get_num_buckets():
    """Return the number of frequency buckets."""
    return len(BUCKET_BOUNDARIES) - 1


def get_learning_rate(iteration: int, total_iterations: int = None) -> float:
    """
    Compute learning rate with log decay.
    
    Args:
        iteration: Current iteration (0-indexed)
        total_iterations: Total number of iterations (defaults to NUM_ITERATIONS)
    
    Returns:
        Learning rate for this iteration
    """
    if total_iterations is None:
        total_iterations = NUM_ITERATIONS
    
    if total_iterations <= 1:
        return LR_INITIAL
    
    progress = iteration / (total_iterations - 1)
    return LR_INITIAL * (LR_FINAL / LR_INITIAL) ** progress
