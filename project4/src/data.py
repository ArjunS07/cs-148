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


def get_val_transform(img_size: int = 512):
    """Deterministic transform for validation."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class ProvidedDigitDataset(Dataset):
    """
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


