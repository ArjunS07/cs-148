"""HF TorchScript pipeline for project 3 (ViT)."""

import argparse

import torch
import torch.nn as nn
from torchvision import transforms

from model import build_model


class DigitClassifierPipeline(nn.Module):

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
        logits = self.model(images)
        return torch.argmax(logits, dim=1)

    @torch.jit.ignore
    def save_pipeline_local(self, path: str):
        self.cpu()
        scripted = torch.jit.script(self)
        scripted.save(path)
        self.to(self.device_)

    @torch.jit.ignore
    def push_to_hub(self, token: str, hf_info: dict):
        import os
        from huggingface_hub import HfApi

        filename = hf_info["filename"]
        repo_id = f"{hf_info['username']}/{hf_info['repo_name']}"

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
        tensor_list = [
            transforms.ToTensor()(img.convert("RGB"))
            for img in pil_images
        ]
        processed = [self.preprocess_layers(t) for t in tensor_list]
        batch = torch.stack(processed).to(self.device_)
        with torch.no_grad():
            predictions = self.forward(batch).tolist()
        return predictions


def load_pipeline(
    checkpoint_path: str,
    use_spt: bool = False,
    use_lsa: bool = False,
    use_dist_token: bool = False,
    device: str = "cpu",
) -> DigitClassifierPipeline:
    model = build_model(use_spt=use_spt, use_lsa=use_lsa, use_dist_token=use_dist_token)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    # Recover SPT/LSA/dist flags from saved args if available
    saved_args = ckpt.get("args", {})
    if saved_args:
        use_spt = saved_args.get("use_spt", use_spt)
        use_lsa = saved_args.get("use_lsa", use_lsa)
        dist_mode = saved_args.get("distillation", "none")
        use_dist_token = (dist_mode == "hard-dist")
        # Rebuild model with correct flags if they differ
        model = build_model(use_spt=use_spt, use_lsa=use_lsa, use_dist_token=use_dist_token)
        model.load_state_dict(ckpt["model_state_dict"])

    pipeline = DigitClassifierPipeline(
        model=model,
        input_height=128,
        input_width=128,
        input_channels=3,
        device=device,
    )
    return pipeline


def save_and_export(pipeline: DigitClassifierPipeline, hf_info: dict):
    try:
        success = pipeline.push_to_hub(token=hf_info["token"], hf_info=hf_info)
        if success:
            import json
            with open("submission.json", "w") as f:
                json.dump(hf_info, f, indent=4)
            return hf_info
        else:
            print("Failed to push pipeline")
            return None
    except Exception as e:
        print(f"Exception: {e}")


USERNAME = "arjuns07"
REPO_NAME = "ee148a-project"
FILENAME = "pipeline-vit.pt"


def main():
    parser = argparse.ArgumentParser(description="Build & export HF pipeline (ViT)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--username", type=str, default=USERNAME)
    parser.add_argument("--token", type=str, default="")
    parser.add_argument("--use-spt", action="store_true", default=False)
    parser.add_argument("--use-lsa", action="store_true", default=False)
    parser.add_argument("--use-dist-token", action="store_true", default=False)
    args = parser.parse_args()

    pipeline = load_pipeline(
        args.checkpoint,
        use_spt=args.use_spt,
        use_lsa=args.use_lsa,
        use_dist_token=args.use_dist_token,
    )
    hf_info = {
        "username": args.username,
        "repo_name": REPO_NAME,
        "filename": FILENAME,
        "checkpoint": args.checkpoint,
        "token": args.token,
    }

    if args.push and args.username and args.token:
        pipeline.push_to_hub(token=args.token, hf_info=hf_info)
        save_and_export(pipeline, hf_info)
    else:
        pipeline.save_pipeline_local(FILENAME)
        print(f"Saved pipeline to {FILENAME}")


if __name__ == "__main__":
    main()
