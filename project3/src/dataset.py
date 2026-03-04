import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from torchvision.datasets import MNIST
from PIL import Image


def get_train_transform(img_size: int = 128):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
        # imagenet values
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


class RepeatedAugSampler(Sampler):
    """Repeats each dataset index N times per epoch with independent shuffles.

    Each epoch, the sampler generates num_repeats independent random permutations
    of the full index set and concatenates them. This means every unique sample is
    seen num_repeats times per epoch, each time with a fresh augmentation draw.

    Usage:
        sampler = RepeatedAugSampler(train_dataset, num_repeats=3)
        loader  = DataLoader(train_dataset, batch_size=64, sampler=sampler, ...)
        # At the start of each epoch call: sampler.set_epoch(epoch)
    """

    def __init__(self, dataset: Dataset, num_repeats: int = 3, seed: int = 0):
        self.n = len(dataset)
        self.num_repeats = num_repeats
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        indices = []
        for rep in range(self.num_repeats):
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch * self.num_repeats + rep)
            indices.extend(torch.randperm(self.n, generator=g).tolist())
        return iter(indices)

    def __len__(self) -> int:
        return self.n * self.num_repeats

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class ProvidedDigitDataset(Dataset):
    """Real training images from the dataset directory.

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


"""
Synthetic MNIST code
"""
def _random_background_crop(bg_paths: list[str], size: int) -> Image.Image:
    path = random.choice(bg_paths)
    bg = Image.open(path).convert("RGB")
    w, h = bg.size
    if w < size or h < size:
        bg = bg.resize((max(w, size), max(h, size)), Image.BILINEAR)
        w, h = bg.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    return bg.crop((x, y, x + size, y + size))


def _render_mnist_digit(mnist_img: Image.Image, size: int = 128) -> Image.Image:
    pad = random.randint(10, 30)
    inner = size - 2 * pad
    digit = mnist_img.resize((inner, inner), Image.BILINEAR)
    digit_np = np.array(digit)

    r = random.randint(50, 255)
    g = random.randint(50, 255)
    b = random.randint(50, 255)

    rgba = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    for y in range(inner):
        for x in range(inner):
            alpha = int(digit_np[y, x])
            if alpha > 20:
                rgba.putpixel((x + pad, y + pad), (r, g, b, alpha))

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
        num_samples: int = 3000,
        img_size: int = 128,
        seed: int = 42,
    ):
        self.transform = transform
        self.img_size = img_size
        self.num_samples = num_samples

        self.mnist = MNIST(
            root=os.path.join(real_data_dir, "..", "mnist_cache"),
            train=True,
            download=True,
        )

        self.bg_paths = [
            os.path.join(real_data_dir, f)
            for f in os.listdir(real_data_dir)
            if f.endswith(".jpg")
        ]

        rng = np.random.RandomState(seed)
        self.mnist_indices = rng.randint(0, len(self.mnist), size=num_samples)
        self.labels = [int(self.mnist.targets[i]) for i in self.mnist_indices]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        mnist_idx = self.mnist_indices[idx]
        mnist_img, label = self.mnist[mnist_idx]

        bg = _random_background_crop(self.bg_paths, self.img_size)
        digit_rgba = _render_mnist_digit(mnist_img, self.img_size)
        bg.paste(digit_rgba, (0, 0), digit_rgba)
        img = bg

        if self.transform is not None:
            img = self.transform(img)
        return img, label, self.SOURCE_SYNTHETIC


class CombinedDataset(Dataset):
    """Concatenation of real + synthetic datasets."""

    def __init__(self, real_dataset: Dataset, synthetic_dataset: Dataset):
        self.real = real_dataset
        self.synthetic = synthetic_dataset

    def __len__(self) -> int:
        return len(self.real) + len(self.synthetic)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        if idx < len(self.real):
            return self.real[idx]
        return self.synthetic[idx - len(self.real)]


def stratified_split(
    data_dir: str, val_fraction: float = 0.15, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split files into train/val with stratification by digit class."""
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".jpg"))

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

    rng.shuffle(train_files)
    rng.shuffle(val_files)
    return train_files, val_files


if __name__ == "__main__":
    data_dir = "../data/dataset"
    train_files, val_files = stratified_split(data_dir)
    print(f"Train: {len(train_files)}, Val: {len(val_files)}")

    from collections import Counter
    train_labels = [int(f.split("_")[-1].replace(".jpg", "").replace("label", "")) for f in train_files]
    val_labels = [int(f.split("_")[-1].replace(".jpg", "").replace("label", "")) for f in val_files]
    print("Train class distribution:", Counter(train_labels))
    print("Val class distribution:", Counter(val_labels))

    ds = ProvidedDigitDataset(data_dir, transform=get_val_transform(128), file_list=train_files[:10])
    img, label, _ = ds[0]
    print(f"Real image shape: {img.shape}, label: {label}")

    sampler = RepeatedAugSampler(ds, num_repeats=3)
    print(f"Sampler length (10 images x  3 repeats): {len(sampler)}")
