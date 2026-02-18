"""HuggingFace pipeline wrapper for ConvNeXt-Femto submission.

Usage:
    python pipeline.py --checkpoint checkpoints/best_model.pt --save pipeline-cnn.pt
    python pipeline.py --checkpoint checkpoints/best_model.pt --push --username YOUR_USERNAME --token YOUR_TOKEN
"""

import argparse

import torch
import torch.nn as nn
from torchvision import transforms

from model import build_model


class DigitClassifierPipeline(nn.Module):
    """
    Accepts variable-resolution RGB input, resizes to 128x128, normalizes,
    and returns predicted class indices.
    """

    def __init__(
        self,
        model: nn.Module,
        input_height: int = 128,
        input_width: int = 128,
        input_channels: int = 3,
        device: str = "cpu",
    ):
        super().__init__()
        self.device_ = torch.device(device)
        self.model = model.to(self.device_)
        self.model.eval()

        self.preprocess_layers = nn.Sequential(
            transforms.Resize((input_height, input_width)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        )

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.preprocess_layers(images)
        logits = self.model(x)
        return torch.argmax(logits, dim=1)

    @torch.jit.ignore
    def save_pipeline_local(self, path: str):
        self.cpu()
        scripted = torch.jit.script(self)
        scripted.save(path)
        self.to(self.device_)

    @torch.jit.ignore
    def push_to_hub(
        self,
        token: str,
        repo_id: str = "ee148a-project",
        filename: str = "pipeline-cnn.pt",
    ):
        import os
        from huggingface_hub import HfApi

        local_path = f"temp_{filename}"
        self.save_pipeline_local(local_path)

        api = HfApi(token=token)
        print(f"Uploading {filename} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload compiled pipeline: {filename}",
        )

        if os.path.exists(local_path):
            os.remove(local_path)

        print(f"Done: https://huggingface.co/{repo_id}/blob/main/{filename}")
        return True

    @torch.jit.ignore
    def run(self, pil_images: list):
        """Run pipeline on a list of PIL images. Returns list of predicted class indices."""
        tensor_list = [
            transforms.ToTensor()(img.convert("RGB"))
            for img in pil_images
        ]
        processed = [self.preprocess_layers(t) for t in tensor_list]
        batch = torch.stack(processed).to(self.device_)
        with torch.no_grad():
            predictions = self.forward(batch).tolist()
        return predictions


def load_pipeline(checkpoint_path: str, model_name: str = "resnet18",
                  device: str = "cpu") -> DigitClassifierPipeline:
    """Load a trained checkpoint into the submission pipeline."""
    model = build_model(model_name)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])

    pipeline = DigitClassifierPipeline(
        model=model,
        input_height=128,
        input_width=128,
        input_channels=3,
        device=device,
    )
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.4f})")
    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Build & export HF pipeline")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--save", type=str, default="pipeline-cnn.pt")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--username", type=str, default="")
    parser.add_argument("--token", type=str, default="")
    args = parser.parse_args()

    pipeline = load_pipeline(args.checkpoint)

    if args.push and args.username and args.token:
        repo_id = f"{args.username}/ee148a-project"
        pipeline.push_to_hub(token=args.token, repo_id=repo_id, filename=args.save)
    else:
        pipeline.save_pipeline_local(args.save)
        print(f"Saved pipeline to {args.save}")


if __name__ == "__main__":
    main()
