"""
Data loading utilities for image datasets.

Supports loading images from directory structures organized by class label
(e.g., digit_0/, digit_1/, etc. or 0/, 1/, etc.)
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Union
import re


def _get_label_from_dirname(dirname: str) -> Optional[int]:
    """
    Extract numeric label from directory name.
    
    Supports formats like: "0", "1", "digit_0", "class_5", etc.
    
    Args:
        dirname: Directory name
    
    Returns:
        Integer label or None if not parseable
    """
    # Try direct integer conversion first
    try:
        return int(dirname)
    except ValueError:
        pass
    
    # Try to find a number in the name
    match = re.search(r'(\d+)', dirname)
    if match:
        return int(match.group(1))
    
    return None


def load_dataset(
    root_dir: Union[str, Path],
    image_size: Optional[int] = None,
    grayscale: bool = True,
    normalize: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Load images from directory structure organized by label.
    
    Expected structure:
        root_dir/
            0/
                image1.png
                image2.png
                ...
            1/
                ...
            ...
    
    Args:
        root_dir: Path containing subdirectories for each class
        image_size: Expected image size (will resize if different). None to keep original.
        grayscale: If True, convert images to grayscale
        normalize: If True, normalize pixel values to [0, 1]
    
    Returns:
        Dict mapping label (int) to array of images with shape (N, H, W) for grayscale
        or (N, H, W, C) for color. Values in [0, 1] if normalized, else [0, 255].
    
    Raises:
        FileNotFoundError: If root_dir doesn't exist
        ValueError: If no valid subdirectories found
    """
    root_dir = Path(root_dir)
    
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")
    
    images_by_label: Dict[int, List[np.ndarray]] = {}
    
    # Supported image extensions
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
    
    # Iterate through subdirectories
    for subdir in sorted(root_dir.iterdir()):
        if not subdir.is_dir():
            continue
        
        label = _get_label_from_dirname(subdir.name)
        if label is None:
            print(f"Warning: Could not parse label from directory '{subdir.name}', skipping")
            continue
        
        images = []
        
        for img_path in sorted(subdir.iterdir()):
            if img_path.suffix.lower() not in extensions:
                continue
            
            try:
                img = Image.open(img_path)
                
                # Convert to grayscale if requested
                if grayscale:
                    img = img.convert('L')
                else:
                    img = img.convert('RGB')
                
                # Resize if needed
                if image_size is not None and (img.width != image_size or img.height != image_size):
                    img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
                
                # Convert to numpy array
                arr = np.array(img, dtype=np.float32 if normalize else np.uint8)
                
                if normalize:
                    arr = arr / 255.0
                
                images.append(arr)
                
            except Exception as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                continue
        
        if images:
            images_by_label[label] = np.stack(images, axis=0)
            print(f"  Loaded {len(images)} images for label {label}")
    
    if not images_by_label:
        raise ValueError(f"No valid image subdirectories found in {root_dir}")
    
    return images_by_label


def save_modified_images(
    images_by_label: Dict[int, np.ndarray],
    output_dir: Union[str, Path],
    prefix: str = "",
    is_normalized: bool = True,
) -> None:
    """
    Save modified images maintaining directory structure.
    
    Creates structure:
        output_dir/
            0/
                {prefix}0.png
                {prefix}1.png
                ...
            1/
                ...
    
    Args:
        images_by_label: Dict mapping label to (N, H, W) array
        output_dir: Base output directory
        prefix: Prefix for image filenames
        is_normalized: If True, assumes values in [0, 1], else [0, 255]
    """
    output_dir = Path(output_dir)
    
    for label, images in images_by_label.items():
        label_dir = output_dir / str(label)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        for i, img in enumerate(images):
            # Convert to uint8
            if is_normalized:
                img_uint8 = np.clip(img * 255, 0, 255).astype(np.uint8)
            else:
                img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
            
            # Save
            img_pil = Image.fromarray(img_uint8, mode='L' if img.ndim == 2 else 'RGB')
            img_path = label_dir / f"{prefix}{i}.png"
            img_pil.save(img_path)
    
    print(f"Saved {sum(len(v) for v in images_by_label.values())} images to {output_dir}")


def get_dataset_info(images_by_label: Dict[int, np.ndarray]) -> dict:
    """
    Get summary information about a loaded dataset.
    
    Args:
        images_by_label: Dict from load_dataset
    
    Returns:
        Dict with keys: num_classes, total_images, images_per_class, image_shape
    """
    labels = sorted(images_by_label.keys())
    sample_images = images_by_label[labels[0]]
    
    return {
        'num_classes': len(labels),
        'labels': labels,
        'total_images': sum(len(v) for v in images_by_label.values()),
        'images_per_class': {k: len(v) for k, v in images_by_label.items()},
        'image_shape': sample_images.shape[1:],  # (H, W) or (H, W, C)
        'dtype': sample_images.dtype,
        'value_range': (sample_images.min(), sample_images.max()),
    }
