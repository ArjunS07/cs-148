"""
Feature extraction using pretrained MobileNetV2.

Extracts deep features from images for computing distribution distances.
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from typing import Dict, Optional, List
from tqdm import tqdm


class FeatureExtractor:
    """
    Extract features from grayscale images using pretrained MobileNetV2.
    
    The model extracts 1280-dimensional features from the penultimate layer,
    which capture high-level visual semantics useful for distribution comparison.
    """
    
    def __init__(
        self,
        device: str = "mps",
        input_size: int = 224,
        batch_size: int = 32,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            device: Computation device ("mps" for Apple Silicon, "cuda", or "cpu")
            input_size: Size to resize images to (MobileNetV2 default is 224)
            batch_size: Batch size for inference
        """
        self.device = self._setup_device(device)
        self.input_size = input_size
        self.batch_size = batch_size
        
        # Load pretrained MobileNetV2
        print(f"Loading MobileNetV2 on {self.device}...")
        self.model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Remove classifier head - we want features from the last conv layer
        # MobileNetV2 architecture: features -> classifier
        # We keep features and remove classifier to get 1280-dim output
        self.model.classifier = nn.Identity()
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Feature dimension for MobileNetV2
        self.feature_dim = 1280
        
        # Preprocessing for ImageNet-pretrained models
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((input_size, input_size)),
            transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    def _setup_device(self, device: str) -> str:
        """Set up and validate the computation device."""
        if device == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            else:
                print("MPS not available, falling back to CPU")
                return "cpu"
        elif device == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            else:
                print("CUDA not available, falling back to CPU")
                return "cpu"
        return "cpu"
    
    def _preprocess_batch(self, images: np.ndarray) -> torch.Tensor:
        """
        Preprocess a batch of images for the model.
        
        Args:
            images: (N, H, W) array of grayscale images, values in [0, 1]
        
        Returns:
            Preprocessed tensor of shape (N, 3, input_size, input_size)
        """
        # Convert to uint8 for ToPILImage
        images_uint8 = (np.clip(images, 0, 1) * 255).astype(np.uint8)
        
        # Preprocess each image
        batch = torch.stack([self.preprocess(img) for img in images_uint8])
        return batch
    
    def extract(
        self,
        images: np.ndarray,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Extract features from a batch of grayscale images.
        
        Args:
            images: (N, H, W) array of grayscale images, values in [0, 1]
            show_progress: If True, show progress bar
        
        Returns:
            (N, 1280) array of features
        """
        n_images = len(images)
        all_features = []
        
        iterator = range(0, n_images, self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting features")
        
        with torch.no_grad():
            for start_idx in iterator:
                end_idx = min(start_idx + self.batch_size, n_images)
                batch_images = images[start_idx:end_idx]
                
                # Preprocess and move to device
                batch = self._preprocess_batch(batch_images)
                batch = batch.to(self.device)
                
                # Extract features
                features = self.model(batch)
                
                # MobileNetV2 outputs (N, 1280, 1, 1) after adaptive avg pool
                # but with classifier removed, we get (N, 1280)
                if features.dim() == 4:
                    features = features.squeeze(-1).squeeze(-1)
                
                all_features.append(features.cpu().numpy())
        
        return np.concatenate(all_features, axis=0)
    
    def extract_all_labels(
        self,
        images_by_label: Dict[int, np.ndarray],
        show_progress: bool = True,
    ) -> Dict[int, np.ndarray]:
        """
        Extract features for all labels in a dataset.
        
        Args:
            images_by_label: Dict mapping label to (N, H, W) array
            show_progress: If True, show progress bar
        
        Returns:
            Dict mapping label to (N, 1280) feature array
        """
        features_by_label = {}
        
        for label in sorted(images_by_label.keys()):
            images = images_by_label[label]
            print(f"  Extracting features for label {label} ({len(images)} images)...")
            features_by_label[label] = self.extract(images, show_progress=False)
        
        return features_by_label


def create_feature_extractor(
    device: Optional[str] = None,
    input_size: int = 224,
) -> FeatureExtractor:
    """
    Factory function to create a feature extractor with sensible defaults.
    
    Args:
        device: Device to use. If None, auto-detect best available.
        input_size: Input size for the model
    
    Returns:
        Configured FeatureExtractor instance
    """
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    return FeatureExtractor(device=device, input_size=input_size)
