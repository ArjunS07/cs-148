"""
Frequency domain manipulation utilities.

Provides tools for creating frequency bucket masks and applying
frequency-domain weights to images.
"""

import numpy as np
from typing import List, Optional, Tuple


def create_bucket_masks(
    image_size: int = 256,
    boundaries: Optional[List[float]] = None,
) -> List[np.ndarray]:
    """
    Create boolean masks for radial frequency buckets.
    
    The frequency spectrum is divided into concentric rings (buckets)
    based on radial distance from the DC component (center).
    
    Args:
        image_size: Size of square image
        boundaries: Radial frequency boundaries (log-spaced recommended).
                   If None, uses default log-spaced boundaries.
    
    Returns:
        List of (image_size, image_size) boolean masks, one per bucket.
        Masks are designed for use with np.fft.fftshift output.
    """
    if boundaries is None:
        # Default log-spaced boundaries for 256x256
        max_freq = np.sqrt(2) * (image_size // 2)
        boundaries = [0, 2, 4, 8, 16, 32, 64, 128, max_freq]
    
    # Create coordinate grid centered at DC
    center = image_size // 2
    y, x = np.ogrid[:image_size, :image_size]
    
    # Compute radial distance from center
    radius = np.sqrt((y - center) ** 2 + (x - center) ** 2)
    
    # Create masks for each bucket
    masks = []
    for i in range(len(boundaries) - 1):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        
        # Bucket includes lower boundary, excludes upper (except for last bucket)
        if i == len(boundaries) - 2:
            mask = (radius >= lower) & (radius <= upper)
        else:
            mask = (radius >= lower) & (radius < upper)
        
        masks.append(mask)
    
    return masks


def apply_frequency_weights(
    image: np.ndarray,
    weights: np.ndarray,
    bucket_masks: List[np.ndarray],
) -> np.ndarray:
    """
    Apply frequency weights to a single image.
    
    This function:
    1. Computes the 2D FFT of the image
    2. Shifts to center the DC component
    3. Multiplies frequency magnitudes by corresponding bucket weights
    4. Shifts back and computes inverse FFT
    5. Returns the real part, clipped to [0, 1]
    
    Args:
        image: (H, W) grayscale image, values in [0, 1]
        weights: (num_buckets,) array of frequency weights
        bucket_masks: Precomputed bucket masks from create_bucket_masks()
    
    Returns:
        Modified (H, W) image, clipped to [0, 1]
    """
    assert len(weights) == len(bucket_masks), \
        f"Number of weights ({len(weights)}) must match number of buckets ({len(bucket_masks)})"
    
    # Compute FFT and shift DC to center
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    
    # Separate magnitude and phase
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # Apply weights to magnitude for each bucket
    modified_magnitude = magnitude.copy()
    for weight, mask in zip(weights, bucket_masks):
        modified_magnitude[mask] *= weight
    
    # Reconstruct FFT with modified magnitude
    modified_fft_shifted = modified_magnitude * np.exp(1j * phase)
    
    # Shift back and inverse FFT
    modified_fft = np.fft.ifftshift(modified_fft_shifted)
    modified_image = np.fft.ifft2(modified_fft)
    
    # Take real part and clip to valid range
    result = np.real(modified_image)
    result = np.clip(result, 0, 1)
    
    return result


def apply_weights_to_batch(
    images: np.ndarray,
    weights: np.ndarray,
    bucket_masks: List[np.ndarray],
) -> np.ndarray:
    """
    Apply the same frequency weights to a batch of images.
    
    Args:
        images: (N, H, W) array of grayscale images
        weights: (num_buckets,) array of frequency weights
        bucket_masks: Precomputed bucket masks
    
    Returns:
        (N, H, W) array of modified images
    """
    return np.stack([
        apply_frequency_weights(img, weights, bucket_masks)
        for img in images
    ], axis=0)


def visualize_bucket_masks(
    bucket_masks: List[np.ndarray],
    boundaries: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Create a visualization of all bucket masks combined.
    
    Each bucket is assigned a different intensity for visualization.
    
    Args:
        bucket_masks: List of boolean masks
        boundaries: Optional boundaries for labeling
    
    Returns:
        (H, W) array with bucket indices as values (0, 1, 2, ...)
    """
    combined = np.zeros_like(bucket_masks[0], dtype=np.float32)
    
    for i, mask in enumerate(bucket_masks):
        combined[mask] = i + 1
    
    return combined


def get_frequency_spectrum(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the frequency spectrum of an image.
    
    Args:
        image: (H, W) grayscale image
    
    Returns:
        Tuple of (magnitude, phase) arrays, both (H, W)
        Magnitude is log-scaled for visualization.
    """
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    
    magnitude = np.abs(fft_shifted)
    phase = np.angle(fft_shifted)
    
    # Log scale magnitude for visualization (add 1 to avoid log(0))
    log_magnitude = np.log1p(magnitude)
    
    return log_magnitude, phase


def compute_radial_power_spectrum(
    image: np.ndarray,
    num_bins: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the radially averaged power spectrum.
    
    Useful for understanding the frequency content distribution.
    
    Args:
        image: (H, W) grayscale image
        num_bins: Number of radial bins
    
    Returns:
        Tuple of (frequencies, power) arrays
    """
    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    power = np.abs(fft_shifted) ** 2
    
    h, w = image.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    radius = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2)
    
    max_radius = np.sqrt(center[0] ** 2 + center[1] ** 2)
    bin_edges = np.linspace(0, max_radius, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    radial_power = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (radius >= bin_edges[i]) & (radius < bin_edges[i + 1])
        if mask.sum() > 0:
            radial_power[i] = power[mask].mean()
    
    return bin_centers, radial_power
