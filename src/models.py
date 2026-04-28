"""
models.py -- CNN architecture families for multi-objective optimization.

Three architecture types, all parameterized by the same search-space variables:
    1. PlainCNN       - Stacked Conv -> BN -> ReLU -> MaxPool blocks
    2. ResidualCNN    - Same structure with learnable skip connections
    3. DepthSepCNN    - MobileNet-style depthwise-separable convolutions

Usage:
    from models import build_model

    model = build_model(
        arch_type="residual",
        input_channels=3,
        input_resolution=32,
        num_classes=10,
        num_conv_layers=3,
        num_channels=64,
        num_fc_units=128,
        dropout_rate=0.3,
    )
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBlock(nn.Module):
    """Standard Conv -> BatchNorm -> ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Two-conv residual block with a learnable 1x1 skip projection.

    Always projects the skip path so that in_ch != out_ch is handled,
    and the projection adds a small amount of learnable capacity even
    when dimensions match.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

        # Always use a 1x1 projection so the skip path is learnable
        self.skip = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1),
            nn.BatchNorm2d(out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + identity)


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise-separable convolution: depthwise 3x3 + pointwise 1x1.

    Produces the same output shape as a standard conv but with far
    fewer parameters (roughly in_ch * 9 + in_ch * out_ch vs in_ch * out_ch * 9).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.bn1(self.depthwise(x)))
        out = self.act(self.bn2(self.pointwise(out)))
        return out


# ---------------------------------------------------------------------------
# Full network architectures
# ---------------------------------------------------------------------------


class PlainCNN(nn.Module):
    """Stack of ConvBlock layers with MaxPool, followed by FC head."""

    def __init__(
        self,
        input_channels: int,
        input_resolution: int,
        num_classes: int,
        num_conv_layers: int,
        num_channels: int,
        num_fc_units: int,
        dropout_rate: float,
    ):
        super().__init__()
        layers = []
        in_ch = input_channels
        spatial = input_resolution

        for i in range(num_conv_layers):
            out_ch = num_channels * (2 ** min(i, 2))  # Double channels up to 3rd layer
            layers.append(ConvBlock(in_ch, out_ch))
            # Pool every layer (but don't pool below 2x2)
            if spatial > 2:
                layers.append(nn.MaxPool2d(2))
                spatial //= 2
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, num_fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(num_fc_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


class ResidualCNN(nn.Module):
    """Stack of ResidualBlocks with MaxPool, followed by FC head."""

    def __init__(
        self,
        input_channels: int,
        input_resolution: int,
        num_classes: int,
        num_conv_layers: int,
        num_channels: int,
        num_fc_units: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Initial conv to project input channels to num_channels
        self.stem = ConvBlock(input_channels, num_channels)

        blocks = []
        in_ch = num_channels
        spatial = input_resolution

        for i in range(num_conv_layers):
            out_ch = num_channels * (2 ** min(i, 2))
            blocks.append(ResidualBlock(in_ch, out_ch))
            if spatial > 2:
                blocks.append(nn.MaxPool2d(2))
                spatial //= 2
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, num_fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(num_fc_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.classifier(x)


class DepthSepCNN(nn.Module):
    """Stack of DepthwiseSeparableBlocks with MaxPool, followed by FC head.

    Much fewer parameters than PlainCNN for the same channel width, making
    this architecture push the compactness frontier of the Pareto front.
    """

    def __init__(
        self,
        input_channels: int,
        input_resolution: int,
        num_classes: int,
        num_conv_layers: int,
        num_channels: int,
        num_fc_units: int,
        dropout_rate: float,
    ):
        super().__init__()
        # Initial standard conv to project to num_channels (depthwise needs >1 channels)
        self.stem = ConvBlock(input_channels, num_channels)

        blocks = []
        in_ch = num_channels
        spatial = input_resolution

        for i in range(num_conv_layers):
            out_ch = num_channels * (2 ** min(i, 2))
            blocks.append(DepthwiseSeparableBlock(in_ch, out_ch))
            if spatial > 2:
                blocks.append(nn.MaxPool2d(2))
                spatial //= 2
            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_ch, num_fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(num_fc_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ARCH_REGISTRY = {
    "plain": PlainCNN,
    "residual": ResidualCNN,
    "depthwise_separable": DepthSepCNN,
}


def build_model(
    arch_type: str,
    input_channels: int,
    input_resolution: int,
    num_classes: int,
    num_conv_layers: int,
    num_channels: int,
    num_fc_units: int,
    dropout_rate: float,
) -> nn.Module:
    """Construct a CNN from the search-space configuration.

    Parameters
    ----------
    arch_type : str
        One of ``"plain"``, ``"residual"``, ``"depthwise_separable"``.
    input_channels : int
        1 for Fashion-MNIST (grayscale), 3 for CIFAR-10 (RGB).
    input_resolution : int
        Spatial size of input images (e.g. 16 or 32).
    num_classes : int
        Number of output classes (10 for both datasets).
    num_conv_layers : int
        Number of convolutional blocks. Search range: [1, 4].
    num_channels : int
        Base channel width. Doubles at each layer (up to 3rd).
        Search range: [8, 128], powers of 2.
    num_fc_units : int
        Width of the FC hidden layer. Search range: [32, 256].
    dropout_rate : float
        Dropout probability before the final classifier. Range: [0.0, 0.5].

    Returns
    -------
    nn.Module
        The constructed CNN, ready for .to(device) and training.
    """
    arch_cls = _ARCH_REGISTRY[arch_type]
    return arch_cls(
        input_channels=input_channels,
        input_resolution=input_resolution,
        num_classes=num_classes,
        num_conv_layers=num_conv_layers,
        num_channels=num_channels,
        num_fc_units=num_fc_units,
        dropout_rate=dropout_rate,
    )


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import DEVICE

    print(f"Device: {DEVICE}\n")

    configs = [
        # (arch, layers, channels, fc, dropout) -- small, medium, large
        ("plain", 2, 16, 64, 0.1),
        ("plain", 4, 128, 256, 0.3),
        ("residual", 2, 16, 64, 0.1),
        ("residual", 4, 128, 256, 0.3),
        ("depthwise_separable", 2, 16, 64, 0.1),
        ("depthwise_separable", 4, 128, 256, 0.3),
    ]

    for arch, layers, ch, fc, drop in configs:
        model = build_model(
            arch_type=arch,
            input_channels=3,
            input_resolution=32,
            num_classes=10,
            num_conv_layers=layers,
            num_channels=ch,
            num_fc_units=fc,
            dropout_rate=drop,
        ).to(DEVICE)

        params = count_parameters(model)
        # Forward pass test
        x = torch.randn(2, 3, 32, 32, device=DEVICE)
        out = model(x)

        print(
            f"{arch:>22s} | L={layers} Ch={ch:>3d} FC={fc:>3d} | "
            f"Params: {params:>8,} | Out: {tuple(out.shape)}"
        )

    print("\n[OK] All architectures built and forward-pass tested.")
