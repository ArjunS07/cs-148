import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

class DigitClassifierPipeline(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        input_height: int,
        input_width: int,
        input_channels: int = 3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()

        # TODO: Define Transforms as a Module
        # Note: We use nn.Sequential so it serializes correctly with TorchScript.
        # Input expected: Tensor (Batch, C, H, W) or (C, H, W)
        self.preprocess_layers = nn.Sequential(
            transforms.Resize((input_height, input_width)), # feel free to remove or change order
            # ... any more you want to add?
        )

        self.input_height = input_height
        self.input_width = input_width
        self.input_channels = input_channels

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        The main entry point for the saved model.
        Args:
            images: Tensor of shape (B, C, H, W) or (B, H, W)
        Returns:
            Tensor of class indices (B,)
        """
        logits = self.model(images)
        predictions = torch.argmax(logits, dim=1)

        return predictions

    @torch.jit.ignore
    def save_pipeline_local(self, path: str):
        """
        Compiles the ENTIRE pipeline (transforms + model + post)
        and saves it to a file.
        """
        self.cpu()
        scripted_model = torch.jit.script(self)
        scripted_model.save(path)
        self.to(self.device)

    @torch.jit.ignore
    def push_to_hub(
        self,
        token: str,
        repo_id: str = 'ee148a-project',
        filename: str = "pipeline-cnn.pt",
    ):
        """
        Saves the pipeline to a local file and pushes it to the Hugging Face Hub.

        Args:
            token (str): HF token.
            repo_id (str): The ID of your repo,
                           e.g., "{username}/ee148a-project"
            filename (str): The name the file will have on the Hub,
                            e.g. 'pipeline-cnn.pt'
        """
        # 1. Save locally first
        local_path = f"temp_{filename}"
        self.save_pipeline_local(local_path)

        # 2. Initialize API
        from huggingface_hub import HfApi, create_repo
        api = HfApi(token=token)

        # 3. Upload the file
        print(f"Uploading {filename} to Hugging Face...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload compiled pipeline: {filename}"
        )

        # 4. Cleanup local temp file
        import os
        if os.path.exists(local_path):
            os.remove(local_path)

        print(f"Success! Upload available at https://huggingface.co/{repo_id}/blob/main/{filename}")
        return True

    @torch.jit.ignore
    def run(self, pil_images: list):
        """Run pipeline on PIL images."""
        if self.input_channels == 3:
            convert_to = 'RGB'
        elif self.input_channels == 1:
            convert_to = 'L'

        tensor_list = [
            transforms.ToTensor()(img.convert(convert_to))
            for img in pil_images
        ]
        processed_tensor_list = [
            self.preprocess_layers(x)
            for x in tensor_list
        ]
        batch = torch.stack(processed_tensor_list).to(self.device)
        predictions = self.forward(batch).tolist()

        return predictions
    

# Test pipeline fuctionality (by running sample images)

def predict_sample(
    pipeline: DigitClassifierPipeline,
    seed: int = None
):
    # Assumes images and labels still exist
    import numpy as np
    if seed is not None:
        np.random.seed(seed)
    random_idxs = np.random.choice(len(images), size=15)
    sample_images = [images[idx] for idx in random_idxs]
    sample_labels = [labels[idx] for idx in random_idxs]
    predictions = pipeline.run(sample_images)
    for img, pred, true in zip(sample_images, predictions, sample_labels):
        if isinstance(pred, (list, tuple, np.ndarray, torch.Tensor)):
            print(f"WARNING! type(pred): {type(pred)}. Ensure a scalar value for each prediction per image")
        print(f"Predicted: {pred}, True: {true}")
        display(img.resize((128, 128)))
        print('='*100)

pipeline = DigitClassifierPipeline(
    model=model,
    input_height= #..,
    input_width= #..,
    input_channels= #..,
)

predict_sample(pipeline)

hf_info = {
    'username': '<your username>',
    'token': '<paste here>',
    'repo_name': 'ee148a-project',   # DON'T CHANGE
    'filename': 'pipeline-cnn.pt'    # DON'T CHANGE
}

def save_and_export(
    pipeline: DigitClassifierPipeline,
    hf_info: dict,
):
    try:
        success = pipeline.push_to_hub(
            token=hf_info['token'],
            repo_id=f"{hf_info['username']}/{hf_info['repo_name']}",
            filename=hf_info['filename']
        )
        if success:
            import json
            with open('submission.json', 'w') as f:
                json.dump(hf_info, f, indent=4)
            print("Saved json to submission.json")
            return hf_info
    except Exception as e:
        print(f"Exception: {e}")


save_and_export(pipeline, hf_info)