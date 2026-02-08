"""
Optimization routines for frequency weight tuning.

Implements gradient estimation via central finite differences and
the main optimization loop with recognizability constraints.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from pathlib import Path
import json

from frequency import apply_weights_to_batch
from metrics import compute_mmd, check_recognizability
from feature_extractor import FeatureExtractor
from visualization import save_iteration_snapshot


@dataclass
class OptimizationState:
    """Tracks optimization progress for visualization and analysis."""
    
    # Per iteration: (num_labels, num_buckets) array
    weights_history: List[np.ndarray] = field(default_factory=list)
    
    # Per iteration: {label: mmd_value}
    dissimilarity_history: List[Dict[int, float]] = field(default_factory=list)
    
    # Per iteration: {label: {ssim, low_freq_corr, template_acc}}
    recognizability_history: List[Dict[int, Dict[str, float]]] = field(default_factory=list)
    
    # Per iteration: {label: bool} - whether update was accepted
    # accepted_updates: List[Dict[int, bool]] = field(default_factory=list)
    
    # Learning rates used
    learning_rates: List[float] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            'weights_history': [w.tolist() for w in self.weights_history],
            'dissimilarity_history': self.dissimilarity_history,
            'recognizability_history': self.recognizability_history,
            # 'accepted_updates': self.accepted_updates,
            'learning_rates': self.learning_rates,
        }
    
    def save(self, filepath: str) -> None:
        """Save state to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def estimate_gradient_central(
    weights: np.ndarray,
    images: np.ndarray,
    gt_features: np.ndarray,
    bucket_masks: List[np.ndarray],
    feature_extractor: FeatureExtractor,
    epsilon: float = 0.05,
    weight_min: float = 0.1,
    weight_max: float = 3.0,
) -> np.ndarray:
    """
    Estimate gradient of dissimilarity w.r.t. weights using central differences.
    
    For each weight w_j:
        gradient[j] ≈ (f(w + ε*e_j) - f(w - ε*e_j)) / (2ε)
    
    where f is the MMD (dissimilarity) and e_j is the j-th unit vector.
    
    Args:
        weights: (num_buckets,) current weights
        images: (N, H, W) images for this label
        gt_features: (M, D) ground truth features for this label
        bucket_masks: Precomputed frequency bucket masks
        feature_extractor: Feature extraction model
        epsilon: Perturbation size for finite differences
        weight_min: Minimum weight value (for clamping perturbations)
        weight_max: Maximum weight value (for clamping perturbations)
    
    Returns:
        (num_buckets,) gradient estimate
    """
    num_buckets = len(weights)
    gradient = np.zeros(num_buckets)
    
    for j in range(num_buckets):
        # Create perturbed weight vectors
        weights_plus = weights.copy()
        weights_minus = weights.copy()
        
        weights_plus[j] = min(weights[j] + epsilon, weight_max)
        weights_minus[j] = max(weights[j] - epsilon, weight_min)
        
        # Actual perturbation size (may be less at boundaries)
        actual_epsilon = (weights_plus[j] - weights_minus[j]) / 2
        
        if actual_epsilon < 1e-8:
            # At boundary, can't compute gradient
            gradient[j] = 0
            continue
        
        # Apply weights and extract features
        modified_plus = apply_weights_to_batch(images, weights_plus, bucket_masks)
        modified_minus = apply_weights_to_batch(images, weights_minus, bucket_masks)
        
        features_plus = feature_extractor.extract(modified_plus)
        features_minus = feature_extractor.extract(modified_minus)
        
        # Compute MMD for both
        mmd_plus = compute_mmd(gt_features, features_plus)
        mmd_minus = compute_mmd(gt_features, features_minus)
        
        # Central difference estimate
        gradient[j] = (mmd_plus - mmd_minus) / (2 * actual_epsilon)
    
    return gradient


def optimize_single_label(
    label: int,
    custom_images: np.ndarray,
    gt_features: np.ndarray,
    prototypes: Dict[int, np.ndarray],
    bucket_masks: List[np.ndarray],
    feature_extractor: FeatureExtractor,
    num_iterations: int = 15,
    lr_initial: float = 0.3,
    lr_final: float = 0.03,
    epsilon: float = 0.05,
    weight_min: float = 0.1,
    weight_max: float = 3.0,
    ssim_threshold: float = 0.6,
    low_freq_threshold: float = 0.9,
    template_threshold: float = 0.9,
    low_freq_cutoff: int = 8,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict]:
    """
    Optimize frequency weights for a single label.
    
    Uses gradient ascent (we want to MAXIMIZE dissimilarity) 
    
    Args:
        label: The label being optimized
        custom_images: (N, H, W) images for this label
        gt_features: (M, D) ground truth features for this label
        prototypes: Dict of all prototypes for template matching
        bucket_masks: Frequency bucket masks
        feature_extractor: Feature extraction model
        num_iterations: Number of optimization iterations
        lr_initial: Initial learning rate
        lr_final: Final learning rate
        epsilon: Perturbation for gradient estimation
        weight_min: Minimum weight value
        weight_max: Maximum weight value
        ssim_threshold: SSIM threshold for recognizability
        low_freq_threshold: Low-frequency correlation threshold
        template_threshold: Template matching threshold
        low_freq_cutoff: Radius for low-frequency mask
        verbose: Print progress
    
    Returns:
        Tuple of (final_weights, history_dict)
    """
    
    num_buckets = len(bucket_masks)
    weights = np.ones(num_buckets)
    labels = np.array([label] * len(custom_images))
    
    history = {
        'dissimilarity': [],
        'recognizability': [],
        # 'accepted': [],
        'weights': [],
    }
    
    for t in range(num_iterations):
        # Compute learning rate with log decay
        if num_iterations > 1:
            progress = t / (num_iterations - 1)
        else:
            progress = 0
        lr = lr_initial * (lr_final / lr_initial) ** progress
        
        # Estimate gradient
        gradient = estimate_gradient_central(
            weights=weights,
            images=custom_images,
            gt_features=gt_features,
            bucket_masks=bucket_masks,
            feature_extractor=feature_extractor,
            epsilon=epsilon,
            weight_min=weight_min,
            weight_max=weight_max,
        )
        
        # Try gradient ascent update (maximize dissimilarity)
        proposed_weights = weights + lr * gradient
        proposed_weights = np.clip(proposed_weights, weight_min, weight_max)
        
        # Check recognizability
        proposed_images = apply_weights_to_batch(custom_images, proposed_weights, bucket_masks)
        passed, metrics = check_recognizability(
            original=custom_images,
            modified=proposed_images,
            labels=labels,
            prototypes=prototypes,
            ssim_threshold=ssim_threshold,
            low_freq_threshold=low_freq_threshold,
            template_threshold=template_threshold,
            low_freq_cutoff=low_freq_cutoff,
        )
        
        # accepted = False
        # if passed:
        #     weights = proposed_weights
        #     accepted = True
        # else:
        #     # Backtrack: try smaller steps
        #     for scale in backtrack_scales:
        #         backtrack_weights = weights + lr * scale * gradient
        #         backtrack_weights = np.clip(backtrack_weights, weight_min, weight_max)
                
        #         backtrack_images = apply_weights_to_batch(
        #             custom_images, backtrack_weights, bucket_masks
        #         )
        #         passed_bt, metrics_bt = check_recognizability(
        #             original=custom_images,
        #             modified=backtrack_images,
        #             labels=labels,
        #             prototypes=prototypes,
        #             ssim_threshold=ssim_threshold,
        #             low_freq_threshold=low_freq_threshold,
        #             template_threshold=template_threshold,
        #             low_freq_cutoff=low_freq_cutoff,
        #         )
                
        #         if passed_bt:
        #             weights = backtrack_weights
        #             metrics = metrics_bt
        #             accepted = True
        #             break
        
        weights = proposed_weights
        # Compute current dissimilarity
        current_images = apply_weights_to_batch(custom_images, weights, bucket_masks)
        current_features = feature_extractor.extract(current_images)
        current_mmd = compute_mmd(gt_features, current_features)
        
        # Record history
        history['dissimilarity'].append(current_mmd)
        history['recognizability'].append(metrics)
        # history['accepted'].append(accepted)
        history['weights'].append(weights.copy())

        
        if verbose:
            # status = "✓" if accepted else "✗"
            print(f"  Iter {t+1:2d}: {weights=} {metrics} MMD={current_mmd:.4f}, lr={lr:.4f}")
    
    return weights, history


def run_optimization(
    custom_images_by_label: Dict[int, np.ndarray],
    gt_features_by_label: Dict[int, np.ndarray],
    prototypes: Dict[int, np.ndarray],
    bucket_masks: List[np.ndarray],
    feature_extractor: FeatureExtractor,
    output_dir: str,
    num_iterations: int = 15,
    lr_initial: float = 0.3,
    lr_final: float = 0.03,
    epsilon: float = 0.05,
    weight_min: float = 0.1,
    weight_max: float = 3.0,
    ssim_threshold: float = 0.6,
    low_freq_threshold: float = 0.9,
    template_threshold: float = 0.9,
    low_freq_cutoff: int = 8,
    save_iterations: bool = True,
) -> Tuple[Dict[int, np.ndarray], OptimizationState]:
    """
    Run full optimization for all labels.
    
    Optimizes each label independently and saves iteration snapshots.
    
    Args:
        custom_images_by_label: Dict mapping label to (N, H, W) images
        gt_features_by_label: Dict mapping label to (M, D) features
        prototypes: Dict mapping label to prototype image
        bucket_masks: Frequency bucket masks
        feature_extractor: Feature extraction model
        output_dir: Directory for saving outputs
        num_iterations: Number of iterations
        lr_initial: Initial learning rate
        lr_final: Final learning rate
        epsilon: Gradient estimation perturbation
        weight_min: Minimum weight
        weight_max: Maximum weight
        ssim_threshold: SSIM threshold
        low_freq_threshold: Low-frequency threshold
        template_threshold: Template matching threshold
        low_freq_cutoff: Low-frequency cutoff radius
        save_iterations: Whether to save iteration snapshots
    
    Returns:
        Tuple of (final_weights_by_label, optimization_state)
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    labels = sorted(custom_images_by_label.keys())
    num_buckets = len(bucket_masks)
    
    # Initialize state
    state = OptimizationState()
    final_weights = {}
    label_histories = {}
    
    # Optimize each label
    for label in labels:
        print(f"\nOptimizing label {label}...")
        
        weights, history = optimize_single_label(
            label=label,
            custom_images=custom_images_by_label[label],
            gt_features=gt_features_by_label[label],
            prototypes=prototypes,
            bucket_masks=bucket_masks,
            feature_extractor=feature_extractor,
            num_iterations=num_iterations,
            lr_initial=lr_initial,
            lr_final=lr_final,
            epsilon=epsilon,
            weight_min=weight_min,
            weight_max=weight_max,
            ssim_threshold=ssim_threshold,
            low_freq_threshold=low_freq_threshold,
            template_threshold=template_threshold,
            low_freq_cutoff=low_freq_cutoff,
        )
        
        final_weights[label] = weights
        label_histories[label] = history
    
    # Aggregate histories into state
    for t in range(num_iterations):
        # Learning rate for this iteration
        if num_iterations > 1:
            progress = t / (num_iterations - 1)
        else:
            progress = 0
        lr = lr_initial * (lr_final / lr_initial) ** progress
        state.learning_rates.append(lr)
        
        # Aggregate weights
        weights_array = np.zeros((len(labels), num_buckets))
        for i, label in enumerate(labels):
            weights_array[i] = label_histories[label]['weights'][t]
        state.weights_history.append(weights_array)
        
        # Aggregate dissimilarity
        dissim_dict = {
            label: label_histories[label]['dissimilarity'][t]
            for label in labels
        }
        state.dissimilarity_history.append(dissim_dict)
        
        # Aggregate recognizability
        recog_dict = {
            label: label_histories[label]['recognizability'][t]
            for label in labels
        }
        state.recognizability_history.append(recog_dict)
        
        # Aggregate accepted updates
        # accepted_dict = {
            # label: label_histories[label]['accepted'][t]
            # for label in labels
        # }
        # state.accepted_updates.append(accepted_dict)
        
        # Save iteration snapshot
        if save_iterations:
            iter_dir = output_dir / "iterations" / f"iter_{t:02d}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            
            # Apply current weights and save images
            modified_by_label = {}
            for label in labels:
                current_weights = label_histories[label]['weights'][t]
                modified_by_label[label] = apply_weights_to_batch(
                    custom_images_by_label[label],
                    current_weights,
                    bucket_masks,
                )
            
            save_iteration_snapshot(modified_by_label, t, str(output_dir))
    
    return final_weights, state


def compute_baseline_dissimilarity(
    custom_images_by_label: Dict[int, np.ndarray],
    gt_features_by_label: Dict[int, np.ndarray],
    feature_extractor: FeatureExtractor,
) -> Dict[int, float]:
    """
    Compute baseline MMD for unmodified custom images.
    
    Args:
        custom_images_by_label: Dict of custom images
        gt_features_by_label: Dict of ground truth features
        feature_extractor: Feature extraction model
    
    Returns:
        Dict mapping label to baseline MMD
    """
    baseline = {}
    
    for label in sorted(custom_images_by_label.keys()):
        custom_features = feature_extractor.extract(custom_images_by_label[label])
        baseline[label] = compute_mmd(gt_features_by_label[label], custom_features)
    
    return baseline
