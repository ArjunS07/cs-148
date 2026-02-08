"""
Visualization utilities for optimization progress and results.

Includes functions for:
- Saving image grids per digit/iteration
- Plotting loss curves
- Visualizing weight evolution
- Before/after comparisons
- Frequency spectrum visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings


# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


def save_digit_grid(
    images: np.ndarray,
    filepath: str,
    title: Optional[str] = None,
    ncols: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> None:
    """
    Save a grid of images for one label/digit.
    
    Args:
        images: (N, H, W) array of grayscale images
        filepath: Where to save the figure
        title: Optional title for the grid
        ncols: Number of columns in grid
        figsize: Figure size (auto-computed if None)
        dpi: Resolution for saved figure
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    
    if figsize is None:
        figsize = (ncols * 2, nrows * 2)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)
    
    for idx in range(nrows * ncols):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]
        
        if idx < n_images:
            ax.imshow(images[idx], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_iteration_snapshot(
    images_by_label: Dict[int, np.ndarray],
    iteration: int,
    output_dir: str,
) -> None:
    """
    Save image grids for all labels at a given iteration.
    
    Creates: output_dir/iterations/iter_XX/digit_Y.png
    
    Args:
        images_by_label: Dict mapping label to (N, H, W) images
        iteration: Current iteration number
        output_dir: Base output directory
    """
    iter_dir = Path(output_dir) / "iterations" / f"iter_{iteration:02d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    
    for label, images in images_by_label.items():
        filepath = iter_dir / f"digit_{label}.png"
        save_digit_grid(
            images=images,
            filepath=str(filepath),
            title=f"Label {label} - Iteration {iteration}",
        )


def plot_loss_curves(
    state,  # OptimizationState
    output_dir: str,
    figsize: Tuple[float, float] = (10, 5),
    dpi: int = 150,
) -> None:
    """
    Plot and save mean dissimilarity curve over iterations.
    
    Args:
        state: OptimizationState object with dissimilarity_history
        output_dir: Directory to save figures
        figsize: Figure size
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract mean dissimilarity per iteration
    iterations = range(len(state.dissimilarity_history))
    mean_dissim = [
        np.mean(list(d.values()))
        for d in state.dissimilarity_history
    ]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(iterations, mean_dissim, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Mean MMD (Dissimilarity)', fontsize=12)
    ax.set_title('Optimization Progress - Mean Dissimilarity', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add improvement annotation
    if len(mean_dissim) > 1:
        improvement = ((mean_dissim[-1] - mean_dissim[0]) / mean_dissim[0]) * 100
        ax.annotate(
            f'Improvement: {improvement:+.1f}%',
            xy=(0.95, 0.05),
            xycoords='axes fraction',
            ha='right',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curves_per_label(
    state,  # OptimizationState
    output_dir: str,
    figsize: Tuple[float, float] = (12, 6),
    dpi: int = 150,
) -> None:
    """
    Plot per-label dissimilarity curves.
    
    Args:
        state: OptimizationState object
        output_dir: Directory to save figures
        figsize: Figure size
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not state.dissimilarity_history:
        return
    
    labels = sorted(state.dissimilarity_history[0].keys())
    iterations = range(len(state.dissimilarity_history))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a colormap for different labels
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))
    
    for label, color in zip(labels, colors):
        dissims = [h[label] for h in state.dissimilarity_history]
        ax.plot(iterations, dissims, '-o', color=color, label=f'Label {label}',
                linewidth=1.5, markersize=4)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('MMD (Dissimilarity)', fontsize=12)
    ax.set_title('Optimization Progress - Per Label', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curves_per_label.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_weight_evolution(
    state,  # OptimizationState
    output_dir: str,
    figsize: Tuple[float, float] = (14, 8),
    dpi: int = 150,
) -> None:
    """
    Plot how weights evolved over iterations.
    
    Creates a heatmap showing weight values for each (label, bucket) over time.
    
    Args:
        state: OptimizationState object
        output_dir: Directory to save figures
        figsize: Figure size
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not state.weights_history:
        return
    
    num_iterations = len(state.weights_history)
    num_labels, num_buckets = state.weights_history[0].shape
    
    # Create subplots: one heatmap per label
    ncols = min(5, num_labels)
    nrows = (num_labels + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes).flatten()
    
    # Custom colormap: blue (< 1) -> white (= 1) -> red (> 1)
    cmap = LinearSegmentedColormap.from_list(
        'weight_cmap',
        [(0, 'blue'), (0.5, 'white'), (1, 'red')]
    )
    
    for label_idx in range(num_labels):
        ax = axes[label_idx]
        
        # Collect weights for this label across iterations
        weights_over_time = np.array([
            state.weights_history[t][label_idx]
            for t in range(num_iterations)
        ])  # Shape: (num_iterations, num_buckets)
        
        im = ax.imshow(
            weights_over_time.T,  # (num_buckets, num_iterations)
            aspect='auto',
            cmap=cmap,
            vmin=0.1,
            vmax=3.0,
        )
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Bucket')
        ax.set_title(f'Label {label_idx}')
        ax.set_xticks(range(0, num_iterations, max(1, num_iterations // 5)))
        ax.set_yticks(range(num_buckets))
    
    # Hide unused subplots
    for idx in range(num_labels, len(axes)):
        axes[idx].axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes[:num_labels], label='Weight', shrink=0.6)
    
    fig.suptitle('Weight Evolution (Blue < 1, White = 1, Red > 1)', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "weight_evolution.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_before_after_comparison(
    original_by_label: Dict[int, np.ndarray],
    modified_by_label: Dict[int, np.ndarray],
    output_dir: str,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> None:
    """
    Create side-by-side comparison of original vs modified images.
    
    Shows one example per label with original, modified, and difference.
    
    Args:
        original_by_label: Dict mapping label to (N, H, W) original images
        modified_by_label: Dict mapping label to (N, H, W) modified images
        output_dir: Directory to save figures
        figsize: Figure size (auto-computed if None)
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = sorted(original_by_label.keys())
    num_labels = len(labels)
    
    if figsize is None:
        figsize = (num_labels * 2, 6)
    
    fig, axes = plt.subplots(3, num_labels, figsize=figsize)
    
    for col, label in enumerate(labels):
        orig = original_by_label[label][0]
        mod = modified_by_label[label][0]
        diff = np.abs(orig - mod)
        
        # Original
        axes[0, col].imshow(orig, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f'{label}')
        axes[0, col].axis('off')
        
        # Modified
        axes[1, col].imshow(mod, cmap='gray', vmin=0, vmax=1)
        axes[1, col].axis('off')
        
        # Difference (amplified for visibility)
        axes[2, col].imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        axes[2, col].axis('off')
    
    # Row labels
    axes[0, 0].set_ylabel('Original', fontsize=12)
    axes[1, 0].set_ylabel('Modified', fontsize=12)
    axes[2, 0].set_ylabel('Difference', fontsize=12)
    
    fig.suptitle('Before / After / Difference', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "before_after_comparison.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_frequency_spectrum_comparison(
    original: np.ndarray,
    modified: np.ndarray,
    output_dir: str,
    label: int = 0,
    figsize: Tuple[float, float] = (12, 4),
    dpi: int = 150,
) -> None:
    """
    Visualize frequency spectrum before/after modification.
    
    Args:
        original: (H, W) original image
        modified: (H, W) modified image
        output_dir: Directory to save figures
        label: Label for filename
        figsize: Figure size
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Original image and spectrum
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    fft_orig = np.fft.fftshift(np.fft.fft2(original))
    mag_orig = np.log1p(np.abs(fft_orig))
    axes[1].imshow(mag_orig, cmap='viridis')
    axes[1].set_title('Original Spectrum')
    axes[1].axis('off')
    
    # Modified image and spectrum
    axes[2].imshow(modified, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title('Modified')
    axes[2].axis('off')
    
    fft_mod = np.fft.fftshift(np.fft.fft2(modified))
    mag_mod = np.log1p(np.abs(fft_mod))
    axes[3].imshow(mag_mod, cmap='viridis')
    axes[3].set_title('Modified Spectrum')
    axes[3].axis('off')
    
    fig.suptitle(f'Frequency Spectrum Comparison - Label {label}', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / f"frequency_comparison_label_{label}.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_bucket_masks(
    bucket_masks: List[np.ndarray],
    boundaries: List[float],
    output_dir: str,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150,
) -> None:
    """
    Visualize frequency bucket masks.
    
    Args:
        bucket_masks: List of boolean masks
        boundaries: Frequency boundaries
        output_dir: Directory to save figures
        figsize: Figure size
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    num_buckets = len(bucket_masks)
    
    if figsize is None:
        figsize = (num_buckets * 2, 2)
    
    fig, axes = plt.subplots(1, num_buckets, figsize=figsize)
    if num_buckets == 1:
        axes = [axes]
    
    for i, (mask, ax) in enumerate(zip(bucket_masks, axes)):
        ax.imshow(mask.astype(float), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'[{boundaries[i]:.0f}, {boundaries[i+1]:.0f})')
        ax.axis('off')
    
    fig.suptitle('Frequency Bucket Masks', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "bucket_masks.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_final_weights(
    weights_by_label: Dict[int, np.ndarray],
    boundaries: List[float],
    output_dir: str,
    figsize: Tuple[float, float] = (10, 6),
    dpi: int = 150,
) -> None:
    """
    Plot final optimized weights as a bar chart.
    
    Args:
        weights_by_label: Dict mapping label to (num_buckets,) weights
        boundaries: Frequency boundaries
        output_dir: Directory to save figures
        figsize: Figure size
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = sorted(weights_by_label.keys())
    num_labels = len(labels)
    num_buckets = len(list(weights_by_label.values())[0])
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(num_buckets)
    width = 0.8 / num_labels
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_labels))
    
    for i, (label, color) in enumerate(zip(labels, colors)):
        offset = (i - num_labels / 2 + 0.5) * width
        ax.bar(x + offset, weights_by_label[label], width, 
               label=f'Label {label}', color=color, alpha=0.8)
    
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Baseline')
    ax.set_xlabel('Frequency Bucket', fontsize=12)
    ax.set_ylabel('Weight', fontsize=12)
    ax.set_title('Final Optimized Weights by Label and Bucket', fontsize=14)
    
    # Create bucket labels
    bucket_labels = [f'{boundaries[i]:.0f}-{boundaries[i+1]:.0f}' 
                     for i in range(num_buckets)]
    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels, rotation=45, ha='right')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "final_weights.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_recognizability_metrics(
    state,  # OptimizationState
    output_dir: str,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 150,
) -> None:
    """
    Plot recognizability metrics over iterations.
    
    Args:
        state: OptimizationState object
        output_dir: Directory to save figures
        figsize: Figure size
        dpi: Resolution
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not state.recognizability_history:
        return
    
    iterations = range(len(state.recognizability_history))
    labels = sorted(state.recognizability_history[0].keys())
    
    # Compute mean metrics across labels
    mean_ssim = []
    mean_low_freq = []
    mean_template = []
    
    for hist in state.recognizability_history:
        mean_ssim.append(np.mean([hist[l]['ssim'] for l in labels]))
        mean_low_freq.append(np.mean([hist[l]['low_freq_corr'] for l in labels]))
    
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # SSIM
    axes[0].plot(iterations, mean_ssim, 'b-o', linewidth=2)
    axes[0].axhline(y=0.6, color='r', linestyle='--', label='Threshold')
    axes[0].set_ylabel('SSIM')
    axes[0].set_title('Recognizability Metrics Over Iterations')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Low-frequency correlation
    axes[1].plot(iterations, mean_low_freq, 'g-o', linewidth=2)
    axes[1].axhline(y=0.9, color='r', linestyle='--', label='Threshold')
    axes[1].set_ylabel('Low-Freq Corr')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Template matching
    axes[2].plot(iterations, mean_template, 'm-o', linewidth=2)
    axes[2].axhline(y=0.9, color='r', linestyle='--', label='Threshold')
    axes[2].set_ylabel('Template Acc')
    axes[2].set_xlabel('Iteration')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "recognizability_metrics.png", dpi=dpi, bbox_inches='tight')
    plt.close(fig)
