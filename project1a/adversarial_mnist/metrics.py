"""
Metrics for measuring dissimilarity and recognizability.

Includes:
- Maximum Mean Discrepancy (MMD) for feature distribution comparison
- SSIM for perceptual similarity
- Low-frequency correlation for structural preservation
- Template matching for digit identity verification
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage


def compute_mmd(
    features1: np.ndarray,
    features2: np.ndarray,
    gamma: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two feature sets.
    
    MMD is a kernel-based distance measure between distributions that
    works well with small sample sizes.
    
    Args:
        features1: (N1, D) array (e.g., ground truth features)
        features2: (N2, D) array (e.g., modified image features)
        gamma: RBF kernel bandwidth. If None, uses median heuristic.
    
    Returns:
        MMD value (higher = more dissimilar distributions)
    """
    if gamma is None:
        # Median heuristic: gamma = 1 / (2 * median_distance^2)
        # Compute on a subsample for efficiency
        combined = np.vstack([features1[:50], features2[:50]])
        pairwise_sq = _pairwise_squared_distances(combined, combined)
        median_dist = np.median(pairwise_sq[pairwise_sq > 0])
        gamma = 1.0 / (2.0 * median_dist + 1e-8)
    
    n1, n2 = len(features1), len(features2)
    
    # Compute kernel matrices
    K11 = _rbf_kernel(features1, features1, gamma)
    K22 = _rbf_kernel(features2, features2, gamma)
    K12 = _rbf_kernel(features1, features2, gamma)
    
    # MMD^2 estimate (unbiased)
    # E[K(X,X')] - 2*E[K(X,Y)] + E[K(Y,Y')]
    # For unbiased estimate, exclude diagonal terms
    
    term1 = (np.sum(K11) - np.trace(K11)) / (n1 * (n1 - 1)) if n1 > 1 else 0
    term2 = (np.sum(K22) - np.trace(K22)) / (n2 * (n2 - 1)) if n2 > 1 else 0
    term3 = 2 * np.mean(K12)
    
    mmd_squared = term1 + term2 - term3
    
    # Return MMD (not squared), handling numerical issues
    return np.sqrt(max(0, mmd_squared))


def _pairwise_squared_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2 * X @ Y.T
    return np.maximum(distances, 0)  # Handle numerical issues


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, gamma: float) -> np.ndarray:
    """Compute RBF (Gaussian) kernel matrix."""
    distances = _pairwise_squared_distances(X, Y)
    return np.exp(-gamma * distances)


def compute_ssim(
    image1: np.ndarray,
    image2: np.ndarray,
    window_size: int = 11,
    k1: float = 0.01,
    k2: float = 0.03,
    data_range: float = 1.0,
) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.
    
    Args:
        image1: (H, W) grayscale image
        image2: (H, W) grayscale image
        window_size: Size of the Gaussian window
        k1, k2: Stability constants
        data_range: Range of pixel values (1.0 for [0,1] images)
    
    Returns:
        SSIM value in [-1, 1], higher = more similar
    """
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    
    # Create Gaussian window
    sigma = window_size / 6.0
    kernel = _gaussian_kernel(window_size, sigma)
    
    # Compute means
    mu1 = ndimage.convolve(image1, kernel, mode='reflect')
    mu2 = ndimage.convolve(image2, kernel, mode='reflect')
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # Compute variances and covariance
    sigma1_sq = ndimage.convolve(image1 ** 2, kernel, mode='reflect') - mu1_sq
    sigma2_sq = ndimage.convolve(image2 ** 2, kernel, mode='reflect') - mu2_sq
    sigma12 = ndimage.convolve(image1 * image2, kernel, mode='reflect') - mu1_mu2
    
    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    
    ssim_map = numerator / (denominator + 1e-8)
    
    return float(np.mean(ssim_map))


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """Create a 2D Gaussian kernel."""
    x = np.arange(size) - size // 2
    kernel_1d = np.exp(-x ** 2 / (2 * sigma ** 2))
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum()


def compute_ssim_batch(
    original: np.ndarray,
    modified: np.ndarray,
) -> float:
    """
    Compute mean SSIM between batches of original and modified images.
    
    Args:
        original: (N, H, W) array of original images
        modified: (N, H, W) array of modified images
    
    Returns:
        Mean SSIM score across all pairs
    """
    scores = [
        compute_ssim(orig, mod)
        for orig, mod in zip(original, modified)
    ]
    return float(np.mean(scores))


def compute_low_freq_correlation(
    original: np.ndarray,
    modified: np.ndarray,
    cutoff_radius: int = 8,
) -> float:
    """
    Compute correlation between low-frequency components of two images.
    
    Low frequencies contain the major shape information that humans
    primarily perceive.
    
    Args:
        original: (H, W) grayscale image
        modified: (H, W) grayscale image
        cutoff_radius: Frequencies below this radius are considered "low"
    
    Returns:
        Correlation coefficient in [-1, 1], higher = more similar
    """
    h, w = original.shape
    center = (h // 2, w // 2)
    
    # Create low-frequency mask
    y, x = np.ogrid[:h, :w]
    mask = ((y - center[0]) ** 2 + (x - center[1]) ** 2) <= cutoff_radius ** 2
    
    # Get FFTs
    fft_orig = np.fft.fftshift(np.fft.fft2(original))
    fft_mod = np.fft.fftshift(np.fft.fft2(modified))
    
    # Extract low-frequency magnitudes
    low_orig = np.abs(fft_orig[mask])
    low_mod = np.abs(fft_mod[mask])
    
    # Compute correlation
    if np.std(low_orig) < 1e-8 or np.std(low_mod) < 1e-8:
        return 1.0  # Both constant = perfectly correlated
    
    correlation = np.corrcoef(low_orig, low_mod)[0, 1]
    
    return float(correlation)


def compute_low_freq_correlation_batch(
    original: np.ndarray,
    modified: np.ndarray,
    cutoff_radius: int = 8,
) -> float:
    """
    Compute mean low-frequency correlation for batches.
    
    Args:
        original: (N, H, W) array
        modified: (N, H, W) array
        cutoff_radius: Low-frequency cutoff
    
    Returns:
        Mean correlation across all pairs
    """
    scores = [
        compute_low_freq_correlation(orig, mod, cutoff_radius)
        for orig, mod in zip(original, modified)
    ]
    return float(np.mean(scores))



def check_recognizability(
    original: np.ndarray,
    modified: np.ndarray,
    labels: np.ndarray,
    prototypes: Dict[int, np.ndarray],
    ssim_threshold: float = 0.6,
    low_freq_threshold: float = 0.9,
    template_threshold: float = 0.9,
    low_freq_cutoff: int = 8,
) -> Tuple[bool, Dict[str, float]]:
    """
    Combined recognizability check using multiple metrics.
    
    An image modification passes if:
    1. SSIM with original >= ssim_threshold
    2. Low-frequency correlation >= low_freq_threshold
    3. Template matching accuracy >= template_threshold
    
    Args:
        original: (N, H, W) array of original images
        modified: (N, H, W) array of modified images
        labels: (N,) array of true labels
        prototypes: Dict mapping label to prototype image
        ssim_threshold: Minimum acceptable SSIM
        low_freq_threshold: Minimum low-frequency correlation
        template_threshold: Minimum template matching accuracy
        low_freq_cutoff: Radius for low-frequency mask
    
    Returns:
        Tuple of (passed: bool, metrics: dict with individual scores)
    """
    # Compute all metrics
    ssim_score = compute_ssim_batch(original, modified)
    low_freq_score = compute_low_freq_correlation_batch(
        original, modified, cutoff_radius=low_freq_cutoff
    )
    
    # Check all thresholds
    passed = (
        ssim_score >= ssim_threshold and
        low_freq_score >= low_freq_threshold
    )
    
    metrics = {
        'ssim': ssim_score,
        'low_freq_corr': low_freq_score,
        'ssim_passed': ssim_score >= ssim_threshold,
        'low_freq_passed': low_freq_score >= low_freq_threshold,
    }
    
    return passed, metrics


def compute_mean_feature_distance(
    features1: np.ndarray,
    features2: np.ndarray,
) -> float:
    """
    Compute L2 distance between mean feature vectors.
    
    A simpler alternative to MMD when sample sizes are very small.
    
    Args:
        features1: (N1, D) array
        features2: (N2, D) array
    
    Returns:
        L2 distance between means
    """
    mu1 = np.mean(features1, axis=0)
    mu2 = np.mean(features2, axis=0)
    return float(np.linalg.norm(mu1 - mu2))
