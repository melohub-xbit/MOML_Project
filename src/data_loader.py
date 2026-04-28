"""
data_loader.py — Data loading and preprocessing for CIFAR-10 and Fashion-MNIST.

Provides unified data loading for the MOO optimization pipeline.
Both datasets are loaded from local files in ./data/ (no internet required).

CUDA-aware defaults:
    - GPU available  → train_subset_size = 20,000 (GPU handles 2x data within budget)
    - CPU only       → train_subset_size = 10,000 (keeps each MOO trial under ~2 min)

Usage:
    from data_loader import get_dataloaders, get_dataset_info, DEVICE

    train_loader, test_loader = get_dataloaders(
        dataset_name="cifar10",
        batch_size=64,
        # train_subset_size auto-selects 20K (GPU) or 10K (CPU) if not specified
        seed=42,
    )
"""

from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Constants & Device Detection
# ---------------------------------------------------------------------------

# Resolve project root from this file's location: src/data_loader.py -> parent.parent
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = str(PROJECT_ROOT / "data")

# Auto-detect CUDA. Used by all project modules for consistent device placement.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CUDA-aware default: 20K samples when GPU accelerates training, 10K on CPU
# to keep each MOO trial under ~2 minutes.
DEFAULT_TRAIN_SUBSET = 20_000 if torch.cuda.is_available() else 10_000

DATASET_INFO = {
    "cifar10": {
        "num_classes": 10,
        "input_channels": 3,
        "default_resolution": 32,
        "class_names": [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck",
        ],
    },
    "fashion_mnist": {
        "num_classes": 10,
        "input_channels": 1,
        "default_resolution": 28,
        "class_names": [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    },
}

# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------


def _build_cifar10_transforms(input_resolution: int = 32):
    """Build train/test transforms for CIFAR-10.

    Applies optional resizing (for the `input_resolution` search-space
    variable), conversion to tensor, and per-channel normalization using
    the CIFAR-10 channel means and standard deviations.
    """
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    )

    transform_list = []
    if input_resolution != 32:
        transform_list.append(transforms.Resize(input_resolution))
    transform_list += [transforms.ToTensor(), normalize]

    # No data-augmentation — we want deterministic, comparable evaluations
    # across trials in the MOO loop.
    train_transform = transforms.Compose(transform_list)
    test_transform = transforms.Compose(transform_list)
    return train_transform, test_transform


def _build_fashion_mnist_transforms(input_resolution: int = 28):
    """Build train/test transforms for Fashion-MNIST.

    Similar to CIFAR-10 but single-channel grayscale.  Uses the
    Fashion-MNIST global mean/std for normalization.
    """
    normalize = transforms.Normalize(mean=[0.2860], std=[0.3530])

    transform_list = []
    if input_resolution != 28:
        transform_list.append(transforms.Resize(input_resolution))
    transform_list += [transforms.ToTensor(), normalize]

    train_transform = transforms.Compose(transform_list)
    test_transform = transforms.Compose(transform_list)
    return train_transform, test_transform


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_dataset_info(dataset_name: str) -> dict:
    """Return metadata dict for the given dataset.

    Keys: num_classes, input_channels, default_resolution, class_names.
    """
    key = dataset_name.lower().replace("-", "_")
    if key not in DATASET_INFO:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Choose from: {list(DATASET_INFO.keys())}"
        )
    return DATASET_INFO[key]


def get_dataloaders(
    dataset_name: str,
    batch_size: int = 64,
    train_subset_size: int | None | str = "auto",
    test_subset_size: int | None = None,
    input_resolution: int | None = None,
    seed: int = 42,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders for the specified dataset.

    Parameters
    ----------
    dataset_name : str
        One of ``"cifar10"`` or ``"fashion_mnist"``.
    batch_size : int
        Mini-batch size used during training.
    train_subset_size : int, None, or "auto"
        Number of training samples to use per trial.
        - ``"auto"`` (default): 20,000 if CUDA GPU is available, else 10,000.
          This keeps each MOO trial within a ~2 min budget.
        - ``int``: use exactly this many samples.
        - ``None``: use the full training set (50K CIFAR / 60K FashionMNIST).
    test_subset_size : int or None
        If set, use a fixed random subset of the test set (e.g. 500
        samples for fast inference-time measurement). ``None`` means
        use the full test set.
    input_resolution : int or None
        Target spatial resolution.  If ``None``, the dataset's native
        resolution is used (32 for CIFAR-10, 28 for Fashion-MNIST).
        Passing 16, for example, will down-sample images to 16×16.
    seed : int
        Random seed for reproducible subset selection.
    num_workers : int
        Number of background data-loading workers.

    Returns
    -------
    train_loader, test_loader : tuple[DataLoader, DataLoader]
    """
    key = dataset_name.lower().replace("-", "_")
    info = get_dataset_info(key)

    if input_resolution is None:
        input_resolution = info["default_resolution"]

    # ---- Resolve "auto" subset size --------------------------------------
    if train_subset_size == "auto":
        train_subset_size = DEFAULT_TRAIN_SUBSET

    # ---- Build transforms ------------------------------------------------
    if key == "cifar10":
        train_tf, test_tf = _build_cifar10_transforms(input_resolution)
    elif key == "fashion_mnist":
        train_tf, test_tf = _build_fashion_mnist_transforms(input_resolution)
    else:
        raise ValueError(f"Unknown dataset: {key}")

    # ---- Load datasets from local files ----------------------------------
    if key == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=DATA_ROOT, train=True, download=False, transform=train_tf,
        )
        test_dataset = datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=False, transform=test_tf,
        )
    else:  # fashion_mnist
        train_dataset = datasets.FashionMNIST(
            root=DATA_ROOT, train=True, download=False, transform=train_tf,
        )
        test_dataset = datasets.FashionMNIST(
            root=DATA_ROOT, train=False, download=False, transform=test_tf,
        )

    # ---- Optionally subsample --------------------------------------------
    rng = np.random.RandomState(seed)

    if train_subset_size is not None and train_subset_size < len(train_dataset):
        indices = rng.choice(len(train_dataset), size=train_subset_size, replace=False)
        train_dataset = Subset(train_dataset, indices.tolist())

    if test_subset_size is not None and test_subset_size < len(test_dataset):
        indices = rng.choice(len(test_dataset), size=test_subset_size, replace=False)
        test_dataset = Subset(test_dataset, indices.tolist())

    # ---- Wrap in DataLoaders ---------------------------------------------
    # Use a generator for reproducible shuffling
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Device         : {DEVICE}")
    print(f"CUDA available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Default subset : {DEFAULT_TRAIN_SUBSET:,} samples")

    for name in ["cifar10", "fashion_mnist"]:
        info = get_dataset_info(name)
        print(f"\n{'='*50}")
        print(f"Dataset: {name}")
        print(f"  Classes     : {info['num_classes']}")
        print(f"  Channels    : {info['input_channels']}")
        print(f"  Resolution  : {info['default_resolution']}x{info['default_resolution']}")

        # Uses CUDA-aware auto default for train_subset_size
        train_loader, test_loader = get_dataloaders(
            dataset_name=name,
            batch_size=64,
            # train_subset_size="auto" → 20K (GPU) or 10K (CPU)
            test_subset_size=500,
            seed=42,
        )

        images, labels = next(iter(train_loader))
        print(f"  Train samples : {len(train_loader.dataset):,}")
        print(f"  Train batches : {len(train_loader)}")
        print(f"  Test  batches : {len(test_loader)}")
        print(f"  Batch shape   : {images.shape}")
        print(f"  Label range   : [{labels.min().item()}, {labels.max().item()}]")
        print(f"  Pixel range   : [{images.min().item():.3f}, {images.max().item():.3f}]")

    print("\n[OK] All datasets loaded successfully from local files.")
