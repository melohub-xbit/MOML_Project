"""
models.py -- CNN architecture for multi-objective optimization.

Single architecture family (PlainCNN). Stack of Conv -> BN -> ReLU -> MaxPool
blocks followed by an FC head. Parameterized by the search-space variables.

Usage:
    from models import build_model

    model = build_model(
        arch_type="plain",
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


# ---------------------------------------------------------------------------
# Full network architecture
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ARCH_REGISTRY = {
    "plain": PlainCNN,
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
        Currently only ``"plain"`` is supported.
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
    num_fc_units : int
        Width of the FC hidden layer.
    dropout_rate : float
        Dropout probability before the final classifier.

    Returns
    -------
    nn.Module
        The constructed CNN, ready for .to(device) and training.
    """
    if arch_type not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unknown arch_type: {arch_type!r}. Supported: {list(_ARCH_REGISTRY)}"
        )
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
        # (layers, channels, fc, dropout) -- small, medium, large
        (1, 8, 128, 0.1),
        (2, 16, 128, 0.2),
        (4, 32, 128, 0.3),
    ]

    for layers, ch, fc, drop in configs:
        model = build_model(
            arch_type="plain",
            input_channels=3,
            input_resolution=32,
            num_classes=10,
            num_conv_layers=layers,
            num_channels=ch,
            num_fc_units=fc,
            dropout_rate=drop,
        ).to(DEVICE)

        params = count_parameters(model)
        x = torch.randn(2, 3, 32, 32, device=DEVICE)
        out = model(x)

        print(
            f"plain | L={layers} Ch={ch:>3d} FC={fc:>3d} | "
            f"Params: {params:>8,} | Out: {tuple(out.shape)}"
        )

    print("\n[OK] PlainCNN built and forward-pass tested.")
