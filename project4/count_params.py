import sys
sys.path.insert(0, "src")

from transformers import CLIPModel, AutoModel
from train_downstream import build_mlp, H1_FOR_DIM


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def print_mlp_info(name, in_dim, dropout):
    model = build_mlp(in_dim, dropout)
    total, _ = count_params(model)
    print(f"{name}  (in_dim={in_dim}, dropout={dropout})")
    for i, layer in enumerate(model):
        print(f"    [{i}] {layer}")
    print(f"  Total parameters: {total:,}")
    print()


# --- Foundation models ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
dino_model = AutoModel.from_pretrained("facebook/dinov2-base")

clip_total, _ = count_params(clip_model)
clip_vision, _ = count_params(clip_model.vision_model)
clip_text, _ = count_params(clip_model.text_model)
clip_other = clip_total - clip_vision - clip_text
dino_total, _ = count_params(dino_model)

print("=" * 55)
print("Foundation model parameter counts")
print("=" * 55)
print(f"CLIP (clip-vit-base-patch32): {clip_total:,} total")
print(f"  Vision encoder: {clip_vision:,}")
print(f"  Text encoder:   {clip_text:,}")
print(f"  Projections:    {clip_other:,}")
print()
print(f"DINOv2 (dinov2-base): {dino_total:,} total")
print()

# --- MLP heads ---
print("=" * 55)
print("Downstream MLP heads (frozen backbone, trained head)")
print("=" * 55)
print_mlp_info("CLIP MLP",                in_dim=512,  dropout=0.3)
print_mlp_info("DINO MLP (CLS only)",     in_dim=768,  dropout=0.2)
print_mlp_info("DINO MLP (CLS+patch mean)", in_dim=1536, dropout=0.2)
