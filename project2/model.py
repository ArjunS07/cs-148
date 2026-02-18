import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    """Stochastic Depth: drop entire residual branch per sample."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # x is always (B, C, H, W) in our usage
        shape = (x.shape[0], 1, 1, 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block.

    depthwise 7x7 conv -> LayerNorm -> 1x1 conv (expand 4x) -> GELU
    -> 1x1 conv (project back) -> layer scale -> drop path -> residual
    """

    def __init__(self, dim: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.layer_scale_init = layer_scale_init
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        x = residual + self.drop_path(x)
        return x


class ConvNeXtFemto(nn.Module):
    """ConvNeXt-Femto: a scaled-down ConvNeXt for small datasets.

    Architecture:
        Patchify stem: 4x4 conv stride 4 -> LayerNorm
        4 stages: depths [2, 2, 6, 2], dims [48, 96, 192, 384]
        Downsample between stages: LayerNorm -> 2x2 conv stride 2
        Head: global avg pool -> LayerNorm -> linear
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 10,
        depths: tuple[int, ...] = (3, 3, 9, 3),
        dims: tuple[int, ...] = (96, 192, 384, 768),
        drop_path_rate: float = 0.1,
        layer_scale_init: float = 1e-6,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Patchify stem: 4x4 conv, stride 4
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )

        # Stochastic depth: linearly increasing drop rates across all blocks
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Build body as a single Sequential: [stage0, down1, stage1, down2, stage2, down3, stage3]
        # This avoids variable-index ModuleList access, which TorchScript can't handle.
        body_layers: list[nn.Module] = []
        block_idx = 0

        for i in range(len(depths)):
            if i > 0:
                body_layers.append(nn.Sequential(
                    LayerNorm2d(dims[i - 1]),
                    nn.Conv2d(dims[i - 1], dims[i], kernel_size=2, stride=2),
                ))

            blocks = []
            for j in range(depths[i]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[i],
                        drop_path=dp_rates[block_idx],
                        layer_scale_init=layer_scale_init,
                    )
                )
                block_idx += 1
            body_layers.append(nn.Sequential(*blocks))

        self.body = nn.Sequential(*body_layers)

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(dims[-1], eps=1e-6),
            nn.Linear(dims[-1], num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.head(x)
        return x



class BasicBlock(nn.Module):
    """Standard ResNet basic block: two 3x3 convs with BatchNorm and skip."""

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        """
        Shape could change:
        1. Stride != 1, so downsampling
        2. Channel dimension changes

        """
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # just use a 1x1 conv - w/ different # of input and output channels, this is a linear transformation
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """ResNet-18 for digit classification.

    Gradual downsampling stem (7x7 stride 2 + maxpool) preserves spatial info
    much better than ConvNeXt's 4x4 stride 4 patchify for training from scratch.

    ~11.2M params with default channels.
    """

    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 10,
        base_width: int = 64,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        w = base_width  # 64 -> layers of [64, 128, 256, 512]

        # Gradual stem: 7x7 stride 2 -> BN -> ReLU -> 3x3 maxpool stride 2
        # 128x128 -> 64x64 -> 32x32
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, w, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # 4 stages, each with 2 BasicBlocks
        # 32x32 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.layer1 = nn.Sequential(
            BasicBlock(w, w, stride=1),
            BasicBlock(w, w, stride=1),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(w, w * 2, stride=2),
            BasicBlock(w * 2, w * 2, stride=1),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(w * 2, w * 4, stride=2),
            BasicBlock(w * 4, w * 4, stride=1),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(w * 4, w * 8, stride=2),
            BasicBlock(w * 8, w * 8, stride=1),
        )

        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w * 8, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(name: str = "resnet18", num_classes: int = 10, **kwargs) -> nn.Module:
    """Build a model by name. Passes extra kwargs to the constructor."""
    if name == "resnet18":
        return ResNet18(num_classes=num_classes, **kwargs)
    elif name == "convnext":
        return ConvNeXtFemto(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ["resnet18", "convnext"]:
        model = build_model(name)
        n = count_parameters(model)
        print(f"{name}: {n:,} params ({n/1e6:.2f}M)")
        for size in [96, 128]:
            x = torch.randn(2, 3, size, size)
            out = model(x)
            print(f"  Input {size}x{size} -> output {out.shape}")
