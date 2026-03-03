
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from torchvision import transforms
from PIL import Image

from dataset import stratified_split

PIPELINE_PATH = "pipeline-cnn.pt"
DATA_DIR = "data/dataset"
N = 0  # images to test (set to 0 for full val set)


def main():
    pipeline = torch.jit.load(PIPELINE_PATH, map_location="cpu")
    pipeline.eval()
    print(f"Loaded {PIPELINE_PATH}")
    print(f"input_height={pipeline.input_height}, input_width={pipeline.input_width}")

    _, val_files = stratified_split(DATA_DIR)
    if N:
        val_files = val_files[:N]
    print(f"Running on {len(val_files)} val images\n")

    to_tensor = transforms.ToTensor()
    correct = 0

    for fname in val_files:
        label = int(fname.split("_")[-1].replace(".jpg", "").replace("label", ""))
        img = Image.open(os.path.join(DATA_DIR, fname)).convert("RGB")

        tensor = to_tensor(img)                          # [0, 1] float
        tensor = pipeline.preprocess_layers(tensor)      # Resize + Normalize
        batch = tensor.unsqueeze(0)

        with torch.no_grad():
            pred = pipeline(batch).item()

        correct += pred == label

    acc = correct / len(val_files)
    print(f"Accuracy: {correct}/{len(val_files)} = {acc:.4f} ({acc*100:.2f}%)")


if __name__ == "__main__":
    main()
