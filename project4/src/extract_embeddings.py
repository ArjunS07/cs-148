"""
Extract and cache CLIP and DINOv2 embeddings for all images.

Extracts one clean embedding per image for both train and val splits.
Also caches CLIP text embeddings for zero-shot evaluation.

Usage:
    cd /path/to/project4
    uv run python src/extract_embeddings.py
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from data import ProvidedDigitDataset

DATA_DIR = "data/dataset"
EMBEDDINGS_DIR = "embeddings"
LOG_PATH = "logs/embedding_extraction.log"
BATCH_SIZE = 32

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

# Prompts using numeral form ("0", "1", ...)
PROMPT_TEMPLATES = [
    'a photo of the number: "{}". ',
    "a handwritten digit {}",
    "the number {} written on paper",
    "an image of the digit {}",
    "a photo of {} written by hand",
    "a close-up of the number {}",
    "the digit {} in a natural setting",
    "a difficult photo of the number {}",
    "a noisy image of the digit {}",
]

# Same prompts using word form ("zero", "one", ...)
PROMPT_TEMPLATES_WORDS = [
    'a photo of the number: "{}". ',
    "a handwritten digit {}",
    "the number {} written on paper",
    "an image of the digit {}",
    "a photo of {} written by hand",
    "a close-up of the number {}",
    "the digit {} in a natural setting",
    "a difficult photo of the number {}",
    "a noisy image of the digit {}",
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")



def extract_clip_embeddings(clip_model, clip_processor, pil_images: list, device: torch.device) -> torch.Tensor:
    """Extract CLIP image embeddings. Returns [N, 512]."""
    inputs = clip_processor(images=pil_images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        vision_out = clip_model.vision_model(pixel_values=pixel_values)
        embeds = clip_model.visual_projection(vision_out.pooler_output)
    return embeds.cpu()


def extract_dino_embeddings(
    dino_model, dino_processor, pil_images: list, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract DINOv2 CLS and CLS+patch-mean embeddings. Returns ([N,768], [N,1536])."""
    inputs = dino_processor(images=pil_images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = dino_model(**inputs)
    last = outputs.last_hidden_state              # [N, T+1, 768]
    cls_token = last[:, 0, :].cpu()               # [N, 768]
    patch_mean = last[:, 1:, :].mean(dim=1).cpu() # [N, 768]
    concat = torch.cat([cls_token, patch_mean], dim=-1)  # [N, 1536]
    return cls_token, concat


def _flush_batch(
    pil_buf: list,
    lbl_buf: list,
    idx_buf: list,
    clip_model,
    clip_processor,
    dino_model,
    dino_processor,
    device: torch.device,
) -> tuple[list, list, list, list, list]:
    """Process a buffer of PIL images and return lists of embedding tensors + metadata."""
    clip_embs = extract_clip_embeddings(clip_model, clip_processor, pil_buf, device)
    dino_cls, dino_concat = extract_dino_embeddings(dino_model, dino_processor, pil_buf, device)
    return (
        list(clip_embs.unbind(0)),
        list(dino_cls.unbind(0)),
        list(dino_concat.unbind(0)),
        lbl_buf[:],
        idx_buf[:],
    )


def compute_text_embeddings(
    clip_model,
    clip_processor,
    device: torch.device,
    templates: list[str],
    class_labels: list[str],
) -> torch.Tensor:
    """
    Compute averaged CLIP text embeddings for each class.

    For each class, fills each template with the class label, L2-normalises each
    template embedding, averages them, then L2-normalises the average.
    Returns [num_classes, 512].
    """
    text_embeds = []
    for label in class_labels:
        prompts = [t.format(label) for t in templates]
        inputs = clip_processor(text=prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            text_out = clip_model.text_model(input_ids=input_ids, attention_mask=attention_mask)
            embeds = clip_model.text_projection(text_out.pooler_output)  # [T, 512]
        embeds = F.normalize(embeds, dim=-1)
        mean_embed = F.normalize(embeds.mean(dim=0), dim=-1)
        text_embeds.append(mean_embed.cpu())
    return torch.stack(text_embeds)  # [num_classes, 512]


def cache_text_embeddings(
    clip_model,
    clip_processor,
    device: torch.device,
    out_path: str,
    templates: list[str] | None = None,
    class_labels: list[str] | None = None,
) -> torch.Tensor:
    """Compute (if not cached) and return text embeddings [10, 512]."""
    if os.path.exists(out_path):
        log.info(f"Text embeddings cached: {out_path}")
        return torch.load(out_path, weights_only=True)

    if templates is None:
        templates = PROMPT_TEMPLATES
    if class_labels is None:
        class_labels = [str(d) for d in range(10)]

    log.info(f"Computing text embeddings ({len(templates)} templates × {len(class_labels)} classes)...")
    text_tensor = compute_text_embeddings(clip_model, clip_processor, device, templates, class_labels)
    torch.save(text_tensor, out_path)
    log.info(f"Saved {text_tensor.shape} → {out_path}")
    return text_tensor


def process_split(
    ds: ProvidedDigitDataset,
    indices: list[int],
    is_train: bool,
    clip_model,
    clip_processor,
    dino_model,
    dino_processor,
    device: torch.device,
) -> tuple[dict, dict, dict]:
    """
    Extract embeddings for a set of dataset indices.

    Returns three dicts (clip, dino_cls, dino_concat), each with keys:
        embeddings [N*n_views, D], labels [N*n_views],
        view_type [N*n_views], image_indices [N*n_views]
    """
    split_tag = "train" if is_train else "val"
    log.info(f"  {split_tag}: {len(indices)} images...")

    all_clip, all_dino_cls, all_dino_concat = [], [], []
    all_labels, all_view_types, all_img_indices = [], [], []

    pil_buffer: list = []
    lbl_buffer: list[int] = []
    idx_buffer: list[int] = []

    for i, orig_idx in enumerate(indices):
        path = os.path.join(DATA_DIR, ds.files[orig_idx])
        pil_img = Image.open(path).convert("RGB")
        pil_buffer.append(pil_img)
        lbl_buffer.append(ds.labels[orig_idx])
        idx_buffer.append(orig_idx)

        if len(pil_buffer) == BATCH_SIZE:
            c, dc, dcc, lbls, idxs = _flush_batch(
                pil_buffer, lbl_buffer, idx_buffer,
                clip_model, clip_processor, dino_model, dino_processor, device,
            )
            all_clip.extend(c)
            all_dino_cls.extend(dc)
            all_dino_concat.extend(dcc)
            all_labels.extend(lbls)
            all_view_types.extend([0] * len(lbls))
            all_img_indices.extend(idxs)
            pil_buffer, lbl_buffer, idx_buffer = [], [], []

        if (i + 1) % 500 == 0:
            log.info(f"    {i + 1}/{len(indices)}")

    if pil_buffer:
        c, dc, dcc, lbls, idxs = _flush_batch(
            pil_buffer, lbl_buffer, idx_buffer,
            clip_model, clip_processor, dino_model, dino_processor, device,
        )
        all_clip.extend(c)
        all_dino_cls.extend(dc)
        all_dino_concat.extend(dcc)
        all_labels.extend(lbls)
        all_view_types.extend([0] * len(lbls))
        all_img_indices.extend(idxs)

    clip_tensor = torch.stack(all_clip)
    dino_cls_tensor = torch.stack(all_dino_cls)
    dino_concat_tensor = torch.stack(all_dino_concat)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    view_type_tensor = torch.tensor(all_view_types, dtype=torch.long)
    img_idx_tensor = torch.tensor(all_img_indices, dtype=torch.long)

    def _make_dict(emb: torch.Tensor) -> dict:
        return {
            "embeddings": emb,
            "labels": labels_tensor,
            "view_type": view_type_tensor,
            "image_indices": img_idx_tensor,
        }

    return _make_dict(clip_tensor), _make_dict(dino_cls_tensor), _make_dict(dino_concat_tensor)


def main():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Load split
    split_path = os.path.join(EMBEDDINGS_DIR, "split_indices.pt")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"{split_path} not found — run split_data.py first")
    split = torch.load(split_path, weights_only=True)
    train_idx: list[int] = split["train_indices"]
    val_idx: list[int] = split["val_indices"]
    log.info(f"Split loaded: {len(train_idx)} train, {len(val_idx)} val")

    ds = ProvidedDigitDataset(DATA_DIR)

    device = get_device()
    log.info(f"Device: {device}")

    from transformers import CLIPModel, CLIPProcessor
    log.info("Loading CLIP B32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    log.info("CLIP loaded.")

    from transformers import AutoImageProcessor, AutoModel
    log.info("Loading DINOv2 Base...")
    dino_model = AutoModel.from_pretrained("facebook/dinov2-base").to(device)
    dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    dino_model.eval()
    for p in dino_model.parameters():
        p.requires_grad = False
    log.info("DINOv2 loaded.")

    # Text embeddings — three variants for zero-shot comparison
    numeral_labels = [str(d) for d in range(10)]
    word_labels = DIGIT_WORDS

    cache_text_embeddings(
        clip_model, clip_processor, device,
        os.path.join(EMBEDDINGS_DIR, "clip_text_embeddings.pt"),
        templates=PROMPT_TEMPLATES, class_labels=numeral_labels,
    )
    cache_text_embeddings(
        clip_model, clip_processor, device,
        os.path.join(EMBEDDINGS_DIR, "clip_text_embeddings_words.pt"),
        templates=PROMPT_TEMPLATES_WORDS, class_labels=word_labels,
    )
    # Combined: average the two normalized per-class embeddings and re-normalize
    combined_path = os.path.join(EMBEDDINGS_DIR, "clip_text_embeddings_combined.pt")
    if not os.path.exists(combined_path):
        emb_num = torch.load(
            os.path.join(EMBEDDINGS_DIR, "clip_text_embeddings.pt"), weights_only=True
        )
        emb_wrd = torch.load(
            os.path.join(EMBEDDINGS_DIR, "clip_text_embeddings_words.pt"), weights_only=True
        )
        combined = F.normalize(emb_num + emb_wrd, dim=-1)  # [10, 512]
        torch.save(combined, combined_path)
        log.info(f"Saved combined text embeddings → {combined_path}")

    # Train embeddings
    train_out_paths = [
        os.path.join(EMBEDDINGS_DIR, "clip_train_embeddings.pt"),
        os.path.join(EMBEDDINGS_DIR, "dino_train_cls.pt"),
        os.path.join(EMBEDDINGS_DIR, "dino_train_concat.pt"),
    ]
    if all(os.path.exists(p) for p in train_out_paths):
        log.info("Train embeddings already cached, skipping.")
    else:
        log.info("Extracting train embeddings...")
        clip_train, dino_cls_train, dino_concat_train = process_split(
            ds, train_idx, is_train=True,
            clip_model=clip_model, clip_processor=clip_processor,
            dino_model=dino_model, dino_processor=dino_processor,
            device=device,
        )
        torch.save(clip_train, train_out_paths[0])
        torch.save(dino_cls_train, train_out_paths[1])
        torch.save(dino_concat_train, train_out_paths[2])
        log.info(f"CLIP train: {clip_train['embeddings'].shape}")
        log.info(f"DINO CLS train: {dino_cls_train['embeddings'].shape}")
        log.info(f"DINO concat train: {dino_concat_train['embeddings'].shape}")

    # Val embeddings
    val_out_paths = [
        os.path.join(EMBEDDINGS_DIR, "clip_val_embeddings.pt"),
        os.path.join(EMBEDDINGS_DIR, "dino_val_cls.pt"),
        os.path.join(EMBEDDINGS_DIR, "dino_val_concat.pt"),
    ]
    if all(os.path.exists(p) for p in val_out_paths):
        log.info("Val embeddings already cached, skipping.")
    else:
        log.info("Extracting val embeddings (1 clean view per image)...")
        clip_val, dino_cls_val, dino_concat_val = process_split(
            ds, val_idx, is_train=False,
            clip_model=clip_model, clip_processor=clip_processor,
            dino_model=dino_model, dino_processor=dino_processor,
            device=device,
        )
        torch.save(clip_val, val_out_paths[0])
        torch.save(dino_cls_val, val_out_paths[1])
        torch.save(dino_concat_val, val_out_paths[2])
        log.info(f"CLIP val: {clip_val['embeddings'].shape}")
        log.info(f"DINO CLS val: {dino_cls_val['embeddings'].shape}")

    log.info("Embedding extraction complete.")


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
