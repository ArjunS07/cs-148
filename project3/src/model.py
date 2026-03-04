"""
Architecture: DeiT-tiny
  depth=12, dim=192, heads=3, patch_size=8, img_size=64
  -> N = (64/8)^2 = 64 patch tokens

Optional extensions (controlled by constructor flags):
  use_spt  — Shifted Patch Tokenization: increases input receptive field at embedding time
  use_lsa  — Locality Self-Attention: diagonal masking + learnable temperature
  use_dist_token — Distillation token (DeiT hard-distillation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Stochastic Depth (drop path)
# ---------------------------------------------------------------------------

class StochasticDepth(nn.Module):
    """Drop an entire residual branch with probability drop_prob during training.

    Applied per-sample in the batch (not the same sample is dropped each time).
    Scales surviving branches by 1/(1-drop_prob) to keep expected value constant.
    """

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Shape: (B, 1, 1) so it broadcasts over sequence and dim dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob


# ---------------------------------------------------------------------------
# Patch Embedding (standard, no SPT)
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Standard patch embedding: split image into non-overlapping patches,
    flatten each patch, then linearly project to dim.

    Input:  (B, 3, H, W)
    Output: (B, N, dim)  where N = (H/patch_size) * (W/patch_size)
    """

    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, dim: int = 192):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)          # (B, dim, n_h, n_w)
        x = x.flatten(2)          # (B, dim, N)
        x = x.transpose(1, 2)     # (B, N, dim)
        return x


# ---------------------------------------------------------------------------
# Shifted Patch Tokenization (SPT)
# ---------------------------------------------------------------------------

class ShiftedPatchTokenization(nn.Module):
    """SPT from https://arxiv.org/abs/2112.13492

    Creates 4 diagonally shifted copies of the image, concatenates them
    channel-wise with the original (3 + 4x 3 = 15 channels total), then
    extracts patches via a linear projection.

    N (number of patch tokens) is unchanged vs. standard tokenization.
    Only the projection input dimensionality changes: 3x Px P -> 15x Px P.

    Input:  (B, 3, H, W)
    Output: (B, N, dim)

    Args:
        shift: Pixel shift distance in each diagonal direction. A shift of 1
               gives each patch 1 border-pixel of information from its 4
               diagonal neighbors. The paper uses shift=1 by default.
    """

    def __init__(self, img_size: int, patch_size: int, dim: int, shift: int = 1):
        super().__init__()
        self.patch_size = patch_size
        self.shift = shift
        in_channels = 3 * 5  # original + 4 diagonal shifts
        self.proj = nn.Linear(in_channels * patch_size * patch_size, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _shift_image(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Translate image content by (dy rows, dx cols) with zero-padding.

        Positive dy  -> content moves down  (top rows zero-padded).
        Positive dx  -> content moves right (left cols zero-padded).
        """
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        src_y0 = max(0, -dy);  src_y1 = H - max(0, dy)
        src_x0 = max(0, -dx);  src_x1 = W - max(0, dx)
        dst_y0 = max(0, dy);   dst_y1 = H - max(0, -dy)
        dst_x0 = max(0, dx);   dst_x1 = W - max(0, -dx)
        out[:, :, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, :, src_y0:src_y1, src_x0:src_x1]
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.shift
        # Four diagonal shifts: up-left, up-right, down-left, down-right
        shifts = [
            self._shift_image(x, -s, -s),
            self._shift_image(x, -s, +s),
            self._shift_image(x, +s, -s),
            self._shift_image(x, +s, +s),
        ]
        x_cat = torch.cat([x] + shifts, dim=1)   # (B, 15, H, W)

        P = self.patch_size
        B, C, H, W = x_cat.shape
        n_h, n_w = H // P, W // P

        # Extract patches via unfold: (B, C, n_h, n_w, P, P)
        patches = x_cat.unfold(2, P, P).unfold(3, P, P)
        # -> (B, n_h*n_w, C*P*P)
        patches = patches.contiguous().view(B, C, n_h * n_w, P * P)
        patches = patches.permute(0, 2, 1, 3).contiguous().view(B, n_h * n_w, C * P * P)

        out = self.proj(patches)    # (B, N, dim)
        out = self.norm(out)
        return out


# ---------------------------------------------------------------------------
# Attention — STUB (implement this)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head self-attention with optional Locality Self-Attention (LSA) modifications.

    LSA adds two things:
      1. Diagonal masking — prevents each token from attending to itself.
         Self-similarity is trivially high and can dominate attention, so masking
         the diagonal forces the model to attend to other patches.
      2. Learnable temperature — replaces the fixed 1/sqrt(head_dim) scaling with
         a per-head learnable scalar, letting the model control attention sharpness.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 3,
        dropout: float = 0.0,
        use_lsa: bool = False,
    ):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.head_dim = dim // heads
        self.use_lsa = use_lsa
        self.drop_prob = dropout

        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

        if use_lsa:
            # One learnable scalar per head; replaces 1/sqrt(head_dim) scaling.
            self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, dim)
        Returns:
            (B, N, dim)
        """
        B, N, C = x.shape

        # Project to Q, K, V in one go and split into (B, heads, N, head_dim) each.
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, heads, N, head_dim)

        if self.use_lsa:
            # Learnable-temperature scaling + diagonal mask.
            scores = (q @ k.transpose(-2, -1)) * self.temperature  # (B, heads, N, N)
            diag = torch.eye(N, dtype=torch.bool, device=x.device)
            scores = scores.masked_fill(diag, float('-inf'))
            attn = self.attn_drop(F.softmax(scores, dim=-1))
            out = attn @ v
        else:
            # F.scaled_dot_product_attention fuses softmax + dropout and supports
            # FlashAttention when available.
            out = F.scaled_dot_product_attention(
                q, k, v, dropout_p=self.drop_prob if self.training else 0.0
            )

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)



# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Position-wise feed-forward network (applied identically to each token).

    Standard Transformer MLP:
      Linear(dim -> hidden_dim) -> GELU -> Dropout -> Linear(hidden_dim -> dim) -> Dropout

    where hidden_dim = int(dim * mlp_ratio), typically 4x the embedding dim.
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Transformer Block (structural skeleton — uses your Attention + MLP)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer block with stochastic depth.

    Structure (Pre-LN, as in DeiT):
      x = x + drop_path(attn(norm1(x)))
      x = x + drop_path(mlp(norm2(x)))
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_lsa: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dropout=dropout, use_lsa=use_lsa)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, dropout=dropout)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# Vision Transformer
# ---------------------------------------------------------------------------

class VisionTransformer(nn.Module):
    """DeiT-style Vision Transformer.

    Default config matches DeiT-tiny adapted for 64x64 input:
      depth=12, dim=192, heads=3, patch_size=8
      -> 64 patch tokens per image

    Optional extensions:
      use_spt       — Shifted Patch Tokenization (5x  input channels for patch embedding)
      use_lsa       — Locality Self-Attention in every block
      use_dist_token — Add a distillation token alongside the CLS token (DeiT hard distillation)
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path_rate: float = 0.1,
        use_spt: bool = False,
        use_lsa: bool = False,
        use_dist_token: bool = False,
    ):
        super().__init__()
        self.use_dist_token = use_dist_token

        # --- Patch embedding ---
        if use_spt:
            self.patch_embed = ShiftedPatchTokenization(img_size, patch_size, dim)
        else:
            self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans=3, dim=dim)

        n_patches = (img_size // patch_size) ** 2

        # --- Learnable tokens ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        if use_dist_token:
            self.dist_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Positional embedding covers: [CLS] (+ [DIST]) + N patch tokens
        n_tokens = n_patches + 1 + (1 if use_dist_token else 0)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, dim))
        self.pos_drop = nn.Dropout(dropout)

        # --- Transformer blocks with linearly increasing stochastic depth ---
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim, heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                use_lsa=use_lsa,
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        # --- Classification head(s) ---
        self.head = nn.Linear(dim, num_classes)
        if use_dist_token:
            self.dist_head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.use_dist_token:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Patch embed -> prepend tokens -> positional embed -> transformer blocks -> norm."""
        B = x.shape[0]
        x = self.patch_embed(x)   # (B, N, dim)

        cls = self.cls_token.expand(B, -1, -1)
        if self.use_dist_token:
            dist = self.dist_token.expand(B, -1, -1)
            x = torch.cat([cls, dist, x], dim=1)
        else:
            x = torch.cat([cls, x], dim=1)

        x = self.pos_drop(x + self.pos_embed)

        for block in self.blocks:
            x = block(x)

        return self.norm(x)   # (B, n_tokens, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inference forward. Always returns a single logit tensor.

        If use_dist_token: returns average of cls_head and dist_head outputs
        (head ensemble as recommended in DeiT paper for inference).
        """
        x = self._forward_features(x)
        cls_out = self.head(x[:, 0])
        if self.use_dist_token:
            dist_out = self.dist_head(x[:, 1])
            return (cls_out + dist_out) / 2.0
        return cls_out

    def forward_train(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Training forward. Returns (cls_logits, dist_logits).

        dist_logits is None when use_dist_token=False.
        Used by train.py so that the distillation loss can be computed on both heads
        separately (see DeiT paper, Section 3.3).
        """
        x = self._forward_features(x)
        cls_out = self.head(x[:, 0])
        if self.use_dist_token:
            dist_out = self.dist_head(x[:, 1])
            return cls_out, dist_out
        return cls_out, None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_model(
    img_size: int = 64,
    patch_size: int = 8,
    num_classes: int = 10,
    dim: int = 192,
    depth: int = 12,
    heads: int = 3,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    drop_path_rate: float = 0.1,
    use_spt: bool = False,
    use_lsa: bool = False,
    use_dist_token: bool = False,
) -> VisionTransformer:
    return VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        num_classes=num_classes,
        dim=dim,
        depth=depth,
        heads=heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        drop_path_rate=drop_path_rate,
        use_spt=use_spt,
        use_lsa=use_lsa,
        use_dist_token=use_dist_token,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for spt, lsa, dist in [(False, False, False), (True, False, False),
                            (False, True, False), (True, True, True)]:
        tag = f"spt={spt} lsa={lsa} dist={dist}"
        m = build_model(use_spt=spt, use_lsa=lsa, use_dist_token=dist)
        n = count_parameters(m)
        print(f"{tag}: {n:,} params ({n/1e6:.2f}M)")
        x = torch.randn(2, 3, 64, 64)
        out = m(x)
        print(f"  inference output: {out.shape}")
        cls_out, dist_out = m.forward_train(x)
        print(f"  train cls_out: {cls_out.shape}, dist_out: {dist_out}")
