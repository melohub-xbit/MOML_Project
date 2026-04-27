"""
train_eval.py -- Training loop and 3-objective evaluation for MOO trials.

Takes a hyperparameter configuration dict, builds the model, trains it,
and returns the three objectives:
    O1: Classification accuracy (maximize)
    O2: Inference time in ms (minimize)
    O3: Trainable parameter count (minimize)

Inference time is ALWAYS measured on CPU with batch_size=1 per the project
spec, ensuring fair hardware-independent comparison even when training on GPU.

Usage:
    from train_eval import train_and_evaluate

    objectives = train_and_evaluate(
        config={
            "arch_type": "residual",
            "num_conv_layers": 3,
            "num_channels": 64,
            "num_fc_units": 128,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "num_epochs": 10,
            "dropout_rate": 0.2,
            "optimizer_type": "Adam",
            "input_resolution": 32,
        },
        dataset_name="cifar10",
        seed=42,
    )
    # objectives = {"accuracy": 0.82, "inference_ms": 1.23, "param_count": 45312}
"""

import time
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from data_loader import get_dataloaders, get_dataset_info, DEVICE, DEFAULT_TRAIN_SUBSET
from models import build_model, count_parameters


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------


def _set_seed(seed: int):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_optimizer(
    model: nn.Module,
    optimizer_type: str,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Create optimizer from search-space config."""
    if optimizer_type == "Adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "SGD":
        return torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")


def _train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int,
    show_progress: bool = True,
):
    """Train the model in-place. No return value."""
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(num_epochs):
        batch_iter = train_loader
        if show_progress:
            batch_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1:02d}/{num_epochs:02d}",
                leave=False,
                dynamic_ncols=True,
                mininterval=0.5,
            )
        for images, labels in batch_iter:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if show_progress:
                batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate_accuracy(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    show_progress: bool = True,
) -> float:
    """Compute top-1 accuracy on the test set. Returns float in [0, 1]."""
    model.eval()
    correct = 0
    total = 0

    eval_iter = test_loader
    if show_progress:
        eval_iter = tqdm(
            test_loader,
            desc="Evaluating",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
        )

    with torch.no_grad():
        for images, labels in eval_iter:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return correct / total


def _measure_inference_time(
    model: nn.Module,
    input_channels: int,
    input_resolution: int,
    num_samples: int = 500,
    warmup: int = 50,
) -> float:
    """Measure average inference time per sample in milliseconds.

    Per project spec: measured on CPU, batch_size=1, torch.no_grad(),
    using time.perf_counter(). Includes warmup passes that are discarded.
    """
    # Always measure on CPU for fair comparison
    cpu_model = model.cpu()
    cpu_model.eval()

    dummy = torch.randn(1, input_channels, input_resolution, input_resolution)

    # Warmup passes (not timed) — lets CPU caches and branch predictors settle
    with torch.no_grad():
        for _ in range(warmup):
            cpu_model(dummy)

    # Timed passes
    times = []
    with torch.no_grad():
        for _ in range(num_samples):
            t0 = time.perf_counter()
            cpu_model(dummy)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    # Move model back to training device
    model.to(DEVICE)

    # Return average in milliseconds
    return float(np.mean(times)) * 1000.0


# ---------------------------------------------------------------------------
# Main entry point for MOO loops
# ---------------------------------------------------------------------------


def train_and_evaluate(
    config: dict,
    dataset_name: str,
    seed: int = 42,
    train_subset_size: int | str = "auto",
    show_progress: bool = True,
    num_workers: int = 2,
) -> dict:
    """Full MOO trial: build model, train, evaluate all 3 objectives.

    Parameters
    ----------
    config : dict
        Hyperparameter configuration with keys:
            arch_type        : str   - "plain", "residual", "depthwise_separable"
            num_conv_layers  : int   - [1, 4]
            num_channels     : int   - [8, 128], powers of 2
            num_fc_units     : int   - [32, 256]
            learning_rate    : float - [1e-5, 1e-2]
            batch_size       : int   - {16, 32, 64}
            num_epochs       : int   - [5, 15]
            dropout_rate     : float - [0.0, 0.5]
            optimizer_type   : str   - "SGD" or "Adam"
            input_resolution : int   - {16, 32}
    dataset_name : str
        "cifar10" or "fashion_mnist"
    seed : int
        Random seed for this trial. For MOO loops, use base_seed + trial_number.
    train_subset_size : int or "auto"
        Training subset size. "auto" = 20K (GPU) / 10K (CPU).
    show_progress : bool
        If True, show tqdm progress bars during train/eval loops.
    num_workers : int
        Number of DataLoader workers. Use 0 in notebooks on Windows to avoid
        multiprocessing issues.

    Returns
    -------
    dict with keys:
        accuracy    : float - Top-1 test accuracy in [0, 1]
        inference_ms: float - Avg ms per sample (CPU, batch=1)
        param_count : int   - Total trainable parameters
    """
    _set_seed(seed)

    # ---- Get dataset metadata --------------------------------------------
    ds_info = get_dataset_info(dataset_name)

    # ---- Build model -----------------------------------------------------
    model = build_model(
        arch_type=config["arch_type"],
        input_channels=ds_info["input_channels"],
        input_resolution=config["input_resolution"],
        num_classes=ds_info["num_classes"],
        num_conv_layers=config["num_conv_layers"],
        num_channels=config["num_channels"],
        num_fc_units=config["num_fc_units"],
        dropout_rate=config["dropout_rate"],
    ).to(DEVICE)

    param_count = count_parameters(model)

    # ---- Load data -------------------------------------------------------
    train_loader, test_loader = get_dataloaders(
        dataset_name=dataset_name,
        batch_size=config["batch_size"],
        train_subset_size=train_subset_size,
        input_resolution=config["input_resolution"],
        seed=seed,
        num_workers=num_workers,
    )

    # ---- Build optimizer -------------------------------------------------
    optimizer = _build_optimizer(model, config["optimizer_type"], config["learning_rate"])

    # ---- Train -----------------------------------------------------------
    _train_model(
        model,
        train_loader,
        optimizer,
        config["num_epochs"],
        show_progress=show_progress,
    )

    # ---- Evaluate objectives ---------------------------------------------
    accuracy = _evaluate_accuracy(model, test_loader, show_progress=show_progress)

    inference_ms = _measure_inference_time(
        model,
        input_channels=ds_info["input_channels"],
        input_resolution=config["input_resolution"],
    )

    return {
        "accuracy": accuracy,
        "inference_ms": inference_ms,
        "param_count": param_count,
    }


# ---------------------------------------------------------------------------
# Sanity check — runs one trial per architecture per dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_configs = [
        {
            "arch_type": "plain",
            "num_conv_layers": 2,
            "num_channels": 16,
            "num_fc_units": 64,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "num_epochs": 3,
            "dropout_rate": 0.1,
            "optimizer_type": "Adam",
            "input_resolution": 32,
        },
        {
            "arch_type": "residual",
            "num_conv_layers": 2,
            "num_channels": 32,
            "num_fc_units": 128,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "num_epochs": 3,
            "dropout_rate": 0.2,
            "optimizer_type": "Adam",
            "input_resolution": 32,
        },
        {
            "arch_type": "depthwise_separable",
            "num_conv_layers": 3,
            "num_channels": 32,
            "num_fc_units": 128,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "num_epochs": 3,
            "dropout_rate": 0.2,
            "optimizer_type": "Adam",
            "input_resolution": 32,
        },
    ]

    print(f"Device: {DEVICE}")
    print(f"Default train subset: {DEFAULT_TRAIN_SUBSET:,}")
    print("=" * 80)

    for dataset in ["cifar10", "fashion_mnist"]:
        print(f"\nDataset: {dataset}")
        print("-" * 80)

        for cfg in test_configs:
            t_start = time.time()
            results = train_and_evaluate(
                config=cfg,
                dataset_name=dataset,
                seed=42,
                train_subset_size=5_000,  # Small subset for quick sanity check
            )
            elapsed = time.time() - t_start

            print(
                f"  {cfg['arch_type']:>22s} | "
                f"Acc: {results['accuracy']:.4f} | "
                f"Infer: {results['inference_ms']:.3f} ms | "
                f"Params: {results['param_count']:>8,} | "
                f"Trial time: {elapsed:.1f}s"
            )

    print("\n[OK] All train-evaluate trials completed successfully.")
