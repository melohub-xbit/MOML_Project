

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
    use_amp: bool = False,
):
    
    criterion = nn.CrossEntropyLoss()
    model.train()

    amp_enabled = bool(use_amp) and torch.cuda.is_available() and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

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
            if amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
    train_subset_size: int | str | None = "auto",
    show_progress: bool = True,
    num_workers: int = 2,
    use_amp: bool = False,
    inference_warmup: int = 50,
    inference_timed: int = 500,
) -> dict:
    
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
        use_amp=use_amp,
    )

    # ---- Evaluate objectives ---------------------------------------------
    accuracy = _evaluate_accuracy(model, test_loader, show_progress=show_progress)

    inference_ms = _measure_inference_time(
        model,
        input_channels=ds_info["input_channels"],
        input_resolution=config["input_resolution"],
        num_samples=inference_timed,
        warmup=inference_warmup,
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
