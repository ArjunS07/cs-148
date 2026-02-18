import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image, ImageDraw, ImageFilter


# ---------------------------------------------------------------------------
# Augmentation transforms
# ---------------------------------------------------------------------------

def get_train_transform(img_size: int = 128):
    """Full augmentation pipeline — no horizontal flips (digit identity)."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        # imagenet values - https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_val_transform(img_size: int = 128):
    """Deterministic transform for validation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])



# ---------------------------------------------------------------------------
# Real-image dataset
# ---------------------------------------------------------------------------

class ProvidedDigitDataset(Dataset):
    """
    Expects flat directory with files named: {id}_img{num}_label{digit}.jpg
    Returns (image, label, source) where source=0 means real.
    """

    SOURCE_REAL = 0

    def __init__(
        self,
        data_dir: str,
        transform=None,
        file_list: list[str] | None = None,
    ):
        self.data_dir = data_dir
        # can be a sequence of stacked transforms
        self.transform = transform

        if file_list is not None:
            self.files = file_list
        else:
            self.files = sorted(
                f for f in os.listdir(data_dir) if f.endswith(".jpg")
            )

        self.labels = []
        for f in self.files:
            label = int(f.split("_")[-1].replace(".jpg", "").replace("label", ""))
            self.labels.append(label)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        path = os.path.join(self.data_dir, self.files[idx])
        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, self.labels[idx], self.SOURCE_REAL


# ---------------------------------------------------------------------------
# Synthetic MNIST supplement
# ---------------------------------------------------------------------------

def _random_background_crop(
    bg_paths: list[str], size: int
) -> Image.Image:
    """Crop a random square patch from a random training image."""
    path = random.choice(bg_paths)
    bg = Image.open(path).convert("RGB")
    w, h = bg.size
    # ensure we can crop
    if w < size or h < size:
        bg = bg.resize((max(w, size), max(h, size)), Image.BILINEAR)
        w, h = bg.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return bg.crop((x, y, x + size, y + size))


def _render_mnist_digit(mnist_img: Image.Image, size: int = 128) -> Image.Image:
    """Render an MNIST digit with random color/thickness onto a transparent canvas."""
    # Resize MNIST 28x28 -> target with some padding variation
    pad = random.randint(10, 30)
    inner = size - 2 * pad
    digit = mnist_img.resize((inner, inner), Image.BILINEAR)
    digit_np = np.array(digit)

    # Random color for the digit stroke
    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)

    # Build RGBA image
    rgba = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for y in range(inner):
        for x in range(inner):
            alpha = int(digit_np[y, x])
            if alpha > 20:
                rgba.putpixel((x + pad, y + pad), (r, g, b, alpha))

    # Slight elastic deformation via random affine
    angle = random.uniform(-10, 10)
    rgba = rgba.rotate(angle, resample=Image.BILINEAR, expand=False)

    return rgba


class SyntheticMNISTDataset(Dataset):
    """MNIST digits rendered onto random background crops from real training images.

    Returns (image, label, source) where source=1 means synthetic.
    """

    SOURCE_SYNTHETIC = 1

    def __init__(
        self,
        real_data_dir: str,
        transform=None,
        num_samples: int = 5000,
        img_size: int = 128,
        seed: int = 42,
    ):
        self.transform = transform
        self.img_size = img_size
        self.num_samples = num_samples

        # Load MNIST
        self.mnist = MNIST(
            root=os.path.join(real_data_dir, "..", "mnist_cache"),
            train=True,
            download=True,
        )

        # Paths to real images for background crops
        self.bg_paths = [
            os.path.join(real_data_dir, f)
            for f in os.listdir(real_data_dir)
            if f.endswith(".jpg")
        ]

        # Pre-select which MNIST samples to use (with replacement)
        rng = np.random.RandomState(seed)
        self.mnist_indices = rng.randint(0, len(self.mnist), size=num_samples)
        self.labels = [int(self.mnist.targets[i]) for i in self.mnist_indices]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        mnist_idx = self.mnist_indices[idx]
        mnist_img, label = self.mnist[mnist_idx]

        # Render digit on background
        bg = _random_background_crop(self.bg_paths, self.img_size)
        digit_rgba = _render_mnist_digit(mnist_img, self.img_size)

        # Composite: paste digit onto background
        bg.paste(digit_rgba, (0, 0), digit_rgba)
        img = bg

        if self.transform is not None:
            img = self.transform(img)

        return img, label, self.SOURCE_SYNTHETIC


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------

class CombinedDataset(Dataset):
    """Concatenation of real + synthetic datasets.

    Returns (image, label, source) where source=0 real, source=1 synthetic.
    """

    def __init__(self, real_dataset: Dataset, synthetic_dataset: Dataset):
        self.real = real_dataset
        self.synthetic = synthetic_dataset

    def __len__(self) -> int:
        return len(self.real) + len(self.synthetic)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        if idx < len(self.real):
            return self.real[idx]
        return self.synthetic[idx - len(self.real)]


def mixup_batch(
    images: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2, num_classes: int = 10
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp to a batch. Returns mixed images and soft label vectors."""
    batch_size = images.size(0)
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(batch_size, device=images.device)

    mixed = lam * images + (1.0 - lam) * images[indices]

    labels_onehot = F.one_hot(labels, num_classes).float()
    labels_shuffled = F.one_hot(labels[indices], num_classes).float()
    soft_labels = lam * labels_onehot + (1.0 - lam) * labels_shuffled

    return mixed, soft_labels


# Train/val split utility

def stratified_split(
    data_dir: str, val_fraction: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split files into train/val with stratification by digit class."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".jpg"))

    # Group by label
    by_label: dict[int, list[str]] = {}
    for f in files:
        label = int(f.split("_")[-1].replace(".jpg", "").replace("label", ""))
        by_label.setdefault(label, []).append(f)

    rng = random.Random(seed)
    train_files, val_files = [], []
    for label in sorted(by_label.keys()):
        label_files = by_label[label]
        rng.shuffle(label_files)
        n_val = max(1, int(len(label_files) * val_fraction))
        val_files.extend(label_files[:n_val])
        train_files.extend(label_files[n_val:])

    # Shuffle both lists so classes aren't grouped together
    rng.shuffle(train_files)
    rng.shuffle(val_files)

    return train_files, val_files


if __name__ == "__main__":
    data_dir = "data/dataset"

    # Test split
    train_files, val_files = stratified_split(data_dir)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    # Test real dataset
    ds = ProvidedDigitDataset(data_dir, transform=get_val_transform(128), file_list=train_files[:100])
    img, label = ds[0]
    print(f"Real image shape: {img.shape}, label: {label}")

    # Test synthetic dataset
    syn = SyntheticMNISTDataset(data_dir, transform=get_val_transform(128), num_samples=50)
    img, label = syn[0]
    print(f"Synthetic image shape: {img.shape}, label: {label}")

    # Test combined
    combined = CombinedDataset(ds, syn)
    print(f"Combined dataset size: {len(combined)}")
