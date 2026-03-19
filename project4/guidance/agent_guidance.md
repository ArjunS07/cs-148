This is guidance for a deep learning class project. The goal is to build a multi(10)-class classifier using foundation models for extremely adversarial digit data, using a dataset of 10,000 examples.

The exact guidelines for this project are in project_instructions.md. A previous project, which is in ../project2/, consisted of building a CNN for this same dataset, which achieved 95% accuracy. The report describing that attempt is in this directory in report2_cnn_mnistw_tex_source.txt.

There are 3 main tasks which need to be accomplished for this project:
1. Zero-shot CLIP classifier
2. CLIP downstream classifier eg. with an MLP classification head that operates on CLIP latents
3. DINO downstream classifier eg. similarly using an MLP head

The starter notebook for this project is in CS148a_proj4_FM_starter.ipynb. I have run the dataset generation code in it already, whose output is loading in data/ right now. `data.py` contains the dataloader used for project 2, where I used synthetic data. For this project, **only use the ProvidedImageDataset loader** (real data) -- do NOT use the synthetic data pipeline.

The starter notebook contains this configuration for CLIP and DINO, which we must use:
```python
from transformers import CLIPProcessor, CLIPModel
import torch

# Loading CLIP
print("Loading CLIP B32...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("CLIP B32 loaded successfully.")

clip_model.eval()
for param in clip_model.parameters():
    param.requires_grad = False


from transformers import AutoImageProcessor, AutoModel

# Loading DINOv2
print("Loading DINOv2 Base...")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base")
dino_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
print("DINOv2 Base loaded successfully.")

dino_model.eval()
for param in dino_model.parameters():
    param.requires_grad = False
```

We will write all our code in src/, store experiment results in checkpoints/, log experiments in logs/. All experiments will be run locally on this device, which is an M5 Macbook Pro with MPS.

---

## Step 1: Data Split

Use an 85:15 stratified train:validation split, matching project 2. For each class, select 15% of samples for validation. Save the split indices to disk so they are reproducible across all experiments.

**Critical rule**: validation set uses ONLY clean (unaugmented) embeddings. Augmented views are never added to the validation set.

---

## Step 2: Cache CLIP Text Embeddings

For each digit (0-9), compute text embeddings for the following 9 prompt templates:
```
"a photo of the number: \"{}\"."
"a handwritten digit {}"
"the number {} written on paper"
"an image of the digit {}"
"a photo of {} written by hand"
"a close-up of the number {}"
"the digit {} in a natural setting"
"a difficult photo of the number {}"
"a noisy image of the digit {}"
```

For each class, L2-normalize each template's text embedding, then average across all 9 templates to produce one text embedding per class. L2-normalize the averaged embedding again. Cache the resulting [10, 512] tensor to disk.

---

## Step 3: Image Augmentation and Embedding Extraction

For each image in the training set, extract **1 clean + 3 augmented = 4 total** embeddings for both CLIP and DINO. For each image in the validation set, extract **only the clean (unaugmented)** embedding.

The augmentation pipeline applied BEFORE the model's processor (operates on PIL images):
```python
def get_augmentation_for_fm():
    """Apply BEFORE the model's processor. Works on PIL images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    ])
```

Note: the crop target is 128 (the original image size). The model's processor handles the upscale to 224 and model-specific normalization. Do NOT apply ToTensor or Normalize manually.

Extraction procedure:
```python
# CLIP embedding extraction:
inputs = clip_processor(images=pil_image, return_tensors="pt")
embedding = clip_model.get_image_features(**inputs)  # shape: [1, 512]

# DINO embedding extraction (two variants):
inputs = dino_processor(images=pil_image, return_tensors="pt")
outputs = dino_model(**inputs)
cls_token = outputs.last_hidden_state[:, 0, :]         # shape: [1, 768]
patch_mean = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # shape: [1, 768]
dino_embedding_cls = cls_token                          # 768-dim
dino_embedding_concat = torch.cat([cls_token, patch_mean], dim=-1)  # 1536-dim
```

Save to disk per model:
- `clip_train_embeddings.pt`: dict with keys `embeddings` [N_train * 4, 512], `labels` [N_train * 4], `view_type` [N_train * 4] (0=clean, 1-3=augmented), `image_indices` [N_train * 4] (index into original dataset, for tracing back to image paths)
- `clip_val_embeddings.pt`: dict with keys `embeddings` [N_val, 512], `labels` [N_val], `image_indices` [N_val]
- Same structure for DINO, but store both CLS-only (768) and CLS+patch-mean (1536) variants

Also save the image paths corresponding to each original dataset index, so we can trace back misclassifications to specific images later.


---

## Step 4: Zero-Shot CLIP Evaluation

Using the cached text embeddings from Step 2 and the **clean** CLIP image embeddings (view_type=0 only from training, plus all val):

```python
logit_scale = clip_model.logit_scale.exp()
img_feats = F.normalize(image_embeddings, dim=-1)
text_feats = F.normalize(text_embeddings, dim=-1)  # [10, 512]
logits = logit_scale * (img_feats @ text_feats.T)
preds = logits.argmax(dim=-1)
```

Evaluate on the full dataset (train + val, using only clean embeddings).

**Save the following artifacts:**
- Overall top-1 accuracy
- Per-class accuracy (10 values)
- Confusion matrix (as both .json data and .png plot)
- List of all misclassified images: (image_path, true_label, predicted_label, confidence)
- Save the top 5 most confidently wrong predictions per class for visualization in the writeup
- Save a grid of misclassified example images as .png for the writeup

---

## Step 5: Downstream Classifier Training

### Architecture

CLIP classifier:
```
Linear(512, 256) → BatchNorm1d(256) → ReLU → Dropout(p) → Linear(256, 128) → BatchNorm1d(128) → ReLU → Dropout(p) → Linear(128, 10)
```

DINO classifier (CLS-only, 768-dim input):
```
Linear(768, 384) → BatchNorm1d(384) → ReLU → Dropout(p) → Linear(384, 128) → BatchNorm1d(128) → ReLU → Dropout(p) → Linear(128, 10)
```

DINO classifier (CLS+patch-mean, 1536-dim input):
```
Linear(1536, 512) → BatchNorm1d(512) → ReLU → Dropout(p) → Linear(512, 128) → BatchNorm1d(128) → ReLU → Dropout(p) → Linear(128, 10)
```

### Loss Function

CrossEntropyLoss with label smoothing = 0.05 (same as project 2).

When embedding-space mixup is applied (see below), compute cross-entropy manually against the interpolated soft labels, same approach as project 2's image-space mixup.

### Embedding-Space Mixup
Based on a boolean flag, during training, with probability p_mix = 0.2, apply mixup to each batch:
```python
lam = np.random.beta(0.2, 0.2)
index = torch.randperm(batch_size)
mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]
mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
```
When mixup is applied, use manual cross-entropy with the soft mixed_labels. When mixup is not applied (80% of batches), use standard CrossEntropyLoss with label smoothing.


### Training-Time Embedding Sampling

For each image in each epoch, **randomly sample 1 of the 4 cached embeddings** (1 clean + 3 augmented). Do NOT feed all 4 in every batch -- that would create near-duplicate embeddings in the same epoch and reduce effective batch diversity.

This means each epoch sees exactly N_train samples (not N_train * 4). The stochasticity comes from seeing different views across epochs.

### Optimizer and Schedule

- AdamW with weight decay 1e-4
- Cosine LR schedule with 5 warmup epochs (linear warmup from 0 to LR_base), then cosine decay to LR_min = 1e-6
- Train for 200 epochs with early stopping (patience = 30 epochs, monitoring val accuracy)
- Batch size 64
- Save best model checkpoint by val accuracy

### Hyperparameter Sweep

Since training is ~seconds per config on precomputed embeddings, run a grid search:

| Hyperparameter | Values |
|---|---|
| Learning rate | 1e-3, 3e-4, 1e-4 |
| Dropout | 0.2, 0.3 |
| Embedding mixup | on, off |
| DINO input | CLS-only (768), CLS+patch-mean (1536) |

For CLIP: 3 LR × 2 dropout × 2 mixup = **12 configs**
For DINO: 3 LR × 2 dropout × 2 mixup × 2 input = **24 configs**

Each config trains for up to 200 epochs with early stopping. Estimated total sweep time: **~15-20 minutes**.

Log all sweep results to a CSV/JSON for later analysis and plotting.

---

## Step 6: Evaluation and Visualization
In each experiment,

### Per-model evaluation (CLIP downstream, DINO downstream):
- Final val accuracy and per-class accuracy
- Confusion matrix (saved as both data and plot)
- List of all misclassified images with (path, true_label, pred_label, confidence)

### Cross-model comparison (required by assignment Part 3):
- Find images correctly classified by CLIP-downstream but misclassified by DINO-downstream, and vice versa
- Save these image lists with predictions from both models
- Select 5-10 representative examples from each direction and save as image grids for the writeup

### Training dynamics (required by assignment Parts 2 and 3):
- Plot training loss curves for the best config and 2-3 comparison configs
- Plot val accuracy curves showing the effect of LR, dropout, mixup choices
- Summary table of sweep results (for inclusion in writeup)
- Show sensitivity to hyperparameter choices with figures

### Zero-shot vs downstream comparison:
- Which images does zero-shot CLIP get right that downstream CLIP gets wrong, and vice versa?
- Save examples for writeup discussion

### Overall model comparison (required by assignment Part 5):
- Per-class accuracy comparison across all models

---

## Step 7: Logging and Artifact Organization

```
checkpoints/
├── clip_zero_shot/
│   ├── results.json          # accuracy, per-class accuracy
│   ├── confusion_matrix.png
│   ├── misclassified.json    # list of (path, true, pred, conf)
│   └── worst_predictions/    # saved images of most confident errors
├── clip_downstream/
│   ├── sweep_results.csv     # all hyperparameter configs and results
│   ├── best_model.pt         # best checkpoint
│   ├── best_config.json      # hyperparams of best model
│   ├── training_curves.json  # loss and accuracy per epoch
│   ├── confusion_matrix.png
│   └── misclassified.json
├── dino_downstream/
│   ├── sweep_results.csv
│   ├── best_model.pt
│   ├── best_config.json
│   ├── training_curves.json
│   ├── confusion_matrix.png
│   └── misclassified.json
└── cross_model/
    ├── clip_right_dino_wrong.json
    ├── dino_right_clip_wrong.json
    └── comparison_examples/   # saved images for writeup

logs/
├── embedding_extraction.log
├── zero_shot_eval.log
├── clip_sweep.log
└── dino_sweep.log

embeddings/
├── split_indices.pt          # train/val indices, reproducible
├── image_paths.json          # mapping from index to image path
├── clip_text_embeddings.pt   # [10, 512] averaged prompt embeddings
├── clip_train_embeddings.pt  # [N_train*4, 512] with metadata
├── clip_val_embeddings.pt    # [N_val, 512]
├── dino_train_cls.pt         # [N_train*4, 768]
├── dino_train_concat.pt      # [N_train*4, 1536]
├── dino_val_cls.pt           # [N_val, 768]
└── dino_val_concat.pt        # [N_val, 1536]
```

---

## Execution Order

1. `src/split_data.py` — Create and save stratified 85:15 split
2. `src/extract_embeddings.py` — Extract and cache all embeddings (~15 min)
3. `src/zero_shot_eval.py` — Run CLIP zero-shot evaluation and save artifacts
4. `src/train_downstream.py` — Run hyperparameter sweep for CLIP and DINO classifiers
5. `src/evaluate.py` — Generate all evaluation artifacts and cross-model comparisons
6. `src/visualize.py` — Generate all plots and figures for writeup

Each script should be independently runnable and idempotent (check for cached results before recomputing). Each script should log to both stdout and the appropriate log file.

---

## Reference: Project 2 Training Setup

Look to `project2_train.py.txt` in this directory for the training loop structure, particularly:
- How mixup is implemented (manual cross-entropy with soft labels)
- Label smoothing setup
- Cosine LR schedule with warmup
- Early stopping logic
- Logging and checkpoint saving