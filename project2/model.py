import torch
import torch.nn as nn
import torch.nn.functional as F

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


def build_model(name: str = "resnet18", num_classes: int = 10, **kwargs) -> nn.Module:
    """Build model, pass extra kwargs to the constructor"""
    if name == "resnet18":
        return ResNet18(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for name in ["resnet18"]:
        model = build_model(name)
        n = count_parameters(model)
        print(f"{name}: {n} params")
        for size in [96, 128]:
            x = torch.randn(2, 3, size, size)
            out = model(x)
            print(f"  Input {size}x{size} -> output {out.shape}")
