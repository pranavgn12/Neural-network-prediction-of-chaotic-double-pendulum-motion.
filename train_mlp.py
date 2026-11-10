#!/usr/bin/env python3
"""Train an MLP to predict double-pendulum angular accelerations.

Supports:
- Multiple trajectories: split by trajectory_id (no trajectory overlap).
- One trajectory (the exact original-simulation dataset): chronological
  80/10/10 train/validation/test split.

Saves:
- Trained checkpoint (.pt)
- Metrics report (.metrics.json)
- Per-epoch history (.history.csv and .history.json)
- Training/validation loss plot
- Learning-rate plot
- Per-target RMSE plot
- Gradient-norm plot
- Actual-vs-predicted scatter plots at selected training epochs
- Final test-set scatter plots and residual plots

Usage:
    python train_mlp.py double_pendulum.csv
    python train_mlp.py double_pendulum.csv --epochs 80 --batch-size 512
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


INPUT_NAMES = [
    "sin(theta1)", "cos(theta1)",
    "sin(theta2)", "cos(theta2)",
    "omega1", "omega2",
]
TARGET_NAMES = ["alpha1", "alpha2"]


class AccelerationMLP(nn.Module):
    """Fully connected network predicting alpha1 and alpha2."""

    def __init__(self, input_size: int = 6, output_size: int = 2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Dataset CSV")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("double_pendulum_mlp.pt"))
    parser.add_argument("--plot-every", type=int, default=10,
                        help="Save validation prediction snapshots every N epochs")
    return parser.parse_args()


def load_dataset(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    rows = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.size == 0:
        raise ValueError("Dataset contains no data rows.")
    rows = np.atleast_1d(rows)

    required = {
        "trajectory_id", "theta1_rad", "theta2_rad",
        "omega1_rad_s", "omega2_rad_s",
        "alpha1_rad_s2", "alpha2_rad_s2",
    }
    available = set(rows.dtype.names or ())
    missing = required - available
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    theta1 = rows["theta1_rad"].astype(np.float64)
    theta2 = rows["theta2_rad"].astype(np.float64)
    omega1 = rows["omega1_rad_s"].astype(np.float64)
    omega2 = rows["omega2_rad_s"].astype(np.float64)

    x = np.column_stack((
        np.sin(theta1), np.cos(theta1),
        np.sin(theta2), np.cos(theta2),
        omega1, omega2,
    )).astype(np.float32)

    y = np.column_stack((
        rows["alpha1_rad_s2"], rows["alpha2_rad_s2"],
    )).astype(np.float32)

    trajectory_ids = rows["trajectory_id"].astype(np.int64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Dataset contains NaN or infinite values.")

    return x, y, trajectory_ids


def make_splits(trajectory_ids: np.ndarray, seed: int):
    """Split by trajectory when possible; otherwise split one trajectory in time."""
    unique_ids = np.unique(trajectory_ids)

    if unique_ids.size >= 10:
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(unique_ids)
        n_train = max(1, int(0.80 * len(shuffled)))
        n_val = max(1, int(0.10 * len(shuffled)))
        if n_train + n_val >= len(shuffled):
            n_train, n_val = len(shuffled) - 2, 1

        train_ids = shuffled[:n_train]
        val_ids = shuffled[n_train:n_train + n_val]
        test_ids = shuffled[n_train + n_val:]

        train_mask = np.isin(trajectory_ids, train_ids)
        val_mask = np.isin(trajectory_ids, val_ids)
        test_mask = np.isin(trajectory_ids, test_ids)
        split_description = "trajectory-wise 80/10/10"
    else:
        # The exact-matching generator emits one continuous trajectory.
        # Preserve CSV time order and use contiguous chronological segments.
        n = len(trajectory_ids)
        if n < 30:
            raise ValueError("Need at least 30 rows for chronological train/val/test split.")
        train_end = int(n * 0.80)
        val_end = int(n * 0.90)
        train_mask = np.zeros(n, dtype=bool)
        val_mask = np.zeros(n, dtype=bool)
        test_mask = np.zeros(n, dtype=bool)
        train_mask[:train_end] = True
        val_mask[train_end:val_end] = True
        test_mask[val_end:] = True
        split_description = "chronological 80/10/10 (single trajectory)"

    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("Train/validation/test split produced an empty partition.")
    return train_mask, val_mask, test_mask, split_description


def make_loader(x, y, batch_size, shuffle):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def evaluate(model, loader, device, y_mean, y_std):
    model.eval()
    losses = []
    predictions = []
    targets = []
    criterion = nn.MSELoss(reduction="sum")

    total = 0
    squared_error = np.zeros(2, dtype=np.float64)
    absolute_error = np.zeros(2, dtype=np.float64)

    for xb, yb in loader:
        xb_device = xb.to(device)
        yb_device = yb.to(device)
        pred_norm = model(xb_device)
        losses.append(float(criterion(pred_norm, yb_device).item()))
        pred = pred_norm.cpu().numpy() * y_std + y_mean
        target = yb.numpy() * y_std + y_mean
        diff = pred - target
        squared_error += np.sum(diff ** 2, axis=0)
        absolute_error += np.sum(np.abs(diff), axis=0)
        total += len(xb)
        predictions.append(pred)
        targets.append(target)

    if total == 0:
        raise ValueError("Cannot evaluate an empty data split.")

    return {
        "normalized_mse": float(sum(losses) / (total * 2)),
        "rmse": np.sqrt(squared_error / total),
        "mae": absolute_error / total,
        "predictions": np.concatenate(predictions),
        "targets": np.concatenate(targets),
    }


def save_scatter(targets, predictions, out_dir: Path, suffix: str):
    for index, name in enumerate(TARGET_NAMES):
        actual = targets[:, index]
        predicted = predictions[:, index]
        low = float(min(actual.min(), predicted.min()))
        high = float(max(actual.max(), predicted.max()))
        if math.isclose(low, high):
            padding = max(abs(low) * 0.05, 1e-6)
            low, high = low - padding, high + padding

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(actual, predicted, s=9, alpha=0.45, label="Samples")
        ax.plot([low, high], [low, high], "k--", linewidth=1.5,
                label="Perfect prediction")
        ax.set_xlabel(f"Actual {name} (rad/s²)")
        ax.set_ylabel(f"Predicted {name} (rad/s²)")
        ax.set_title(f"{name}: actual vs predicted ({suffix})")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_scatter_{suffix}.png", dpi=180)
        plt.close(fig)


def save_residual_plots(targets, predictions, out_dir: Path):
    for index, name in enumerate(TARGET_NAMES):
        residual = targets[:, index] - predictions[:, index]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(residual, bins=50, edgecolor="black", alpha=0.8)
        ax.axvline(0.0, color="black", linestyle="--")
        ax.set_xlabel(f"Actual - predicted {name} (rad/s²)")
        ax.set_ylabel("Sample count")
        ax.set_title(f"{name}: test residual distribution")
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_residual_histogram.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size and patience must be positive.")
    if args.lr <= 0 or args.plot_every < 1:
        raise ValueError("learning rate and plot-every must be positive.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    x_raw, y_raw, trajectory_ids = load_dataset(args.csv)
    train_mask, val_mask, test_mask, split_description = make_splits(
        trajectory_ids, args.seed
    )

    # Fit normalization only on the training partition.
    x_mean = x_raw[train_mask].mean(axis=0)
    x_std = x_raw[train_mask].std(axis=0)
    y_mean = y_raw[train_mask].mean(axis=0)
    y_std = y_raw[train_mask].std(axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_std[y_std < 1e-8] = 1.0

    x = ((x_raw - x_mean) / x_std).astype(np.float32)
    y = ((y_raw - y_mean) / y_std).astype(np.float32)

    train_loader = make_loader(x[train_mask], y[train_mask], args.batch_size, True)
    train_eval_loader = make_loader(x[train_mask], y[train_mask], args.batch_size, False)
    val_loader = make_loader(x[val_mask], y[val_mask], args.batch_size, False)
    test_loader = make_loader(x[test_mask], y[test_mask], args.batch_size, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AccelerationMLP().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4
    )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_dir = output_path.parent / f"{output_path.stem}_training_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rows: {len(x_raw):,}")
    print(f"Split: {split_description}")
    print(f"Samples: train={int(train_mask.sum()):,}, "
          f"validation={int(val_mask.sum()):,}, test={int(test_mask.sum()):,}")
    print(f"Device: {device}")
    if np.unique(trajectory_ids).size < 10:
        print("NOTE: This dataset contains one/few initial-condition trajectory(s); "
              "chronological test results do not demonstrate generalization to "
              "unseen initial conditions.")

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0
    snapshot_epochs = {1, 10, 25}
    snapshots = {}

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        batch_grad_norms = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = criterion(prediction, yb)
            loss.backward()

            grad_sq_sum = 0.0
            for parameter in model.parameters():
                if parameter.grad is not None:
                    grad_sq_sum += float(torch.sum(parameter.grad.detach() ** 2).item())
            batch_grad_norms.append(math.sqrt(grad_sq_sum))

            optimizer.step()
            loss_sum += float(loss.item()) * len(xb)
            sample_count += len(xb)

        train_loss = loss_sum / max(sample_count, 1)
        train_eval = evaluate(model, train_eval_loader, device, y_mean, y_std)
        val_eval = evaluate(model, val_loader, device, y_mean, y_std)
        val_loss = val_eval["normalized_mse"]
        current_lr = float(optimizer.param_groups[0]["lr"])
        mean_grad_norm = float(np.mean(batch_grad_norms)) if batch_grad_norms else 0.0

        history.append({
            "epoch": epoch,
            "train_mse_normalized": train_loss,
            "validation_mse_normalized": val_loss,
            "train_rmse_alpha1_rad_s2": float(train_eval["rmse"][0]),
            "train_rmse_alpha2_rad_s2": float(train_eval["rmse"][1]),
            "validation_rmse_alpha1_rad_s2": float(val_eval["rmse"][0]),
            "validation_rmse_alpha2_rad_s2": float(val_eval["rmse"][1]),
            "learning_rate": current_lr,
            "gradient_norm": mean_grad_norm,
        })

        if epoch in snapshot_epochs or epoch % args.plot_every == 0:
            snapshots[epoch] = (
                val_eval["targets"].copy(),
                val_eval["predictions"].copy(),
            )

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train MSE={train_loss:.6g} | val MSE={val_loss:.6g} | "
            f"val RMSE=({val_eval['rmse'][0]:.5g}, {val_eval['rmse'][1]:.5g}) "
            f"rad/s² | grad={mean_grad_norm:.4g} | lr={current_lr:.2e}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    model.eval()

    test_eval = evaluate(model, test_loader, device, y_mean, y_std)
    test_pred = test_eval["predictions"]
    test_target = test_eval["targets"]
    test_rmse = test_eval["rmse"]
    test_mae = test_eval["mae"]

    # Save model checkpoint with the same keys expected by the demo app.
    checkpoint = {
        # Keep the model on its training device for the remaining evaluation/plots.
        # Store CPU copies in the checkpoint without moving the live model.
        "model_state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "input_names": INPUT_NAMES,
        "target_names": TARGET_NAMES,
        "best_epoch": best_epoch,
        "validation_mse_normalized": float(best_val),
        "test_rmse_rad_s2": test_rmse.tolist(),
        "test_mae_rad_s2": test_mae.tolist(),
        "architecture": "MLP(6, 128, 128, 64, 2), GELU",
        "dataset": str(args.csv),
        "split": split_description,
    }
    torch.save(checkpoint, output_path)

    # Per-epoch machine-readable history.
    history_json = output_path.with_suffix(".history.json")
    history_csv = output_path.with_suffix(".history.csv")
    history_json.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    # 1. Training and validation loss.
    epochs = [item["epoch"] for item in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [item["train_mse_normalized"] for item in history],
            label="Training MSE")
    ax.plot(epochs, [item["validation_mse_normalized"] for item in history],
            label="Validation MSE")
    ax.axvline(best_epoch, color="gray", linestyle="--",
               label=f"Best epoch: {best_epoch}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Normalized MSE")
    ax.set_title("Training and validation loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_training_validation_loss.png", dpi=180)
    plt.close(fig)

    # 2. Learning rate.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [item["learning_rate"] for item in history], marker=".")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate (log scale)")
    ax.set_title("Learning rate schedule")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "02_learning_rate.png", dpi=180)
    plt.close(fig)

    # 3. Per-output train/validation RMSE.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, name in enumerate(TARGET_NAMES):
        axes[index].plot(epochs, [item[f"train_rmse_{name}_rad_s2"] for item in history],
                         label="Train RMSE")
        axes[index].plot(epochs, [item[f"validation_rmse_{name}_rad_s2"] for item in history],
                         label="Validation RMSE")
        axes[index].set_title(f"{name} RMSE")
        axes[index].set_xlabel("Epoch")
        axes[index].set_ylabel("RMSE (rad/s²)")
        axes[index].grid(True, alpha=0.25)
        axes[index].legend()
    fig.suptitle("Angular-acceleration error during training")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_acceleration_rmse.png", dpi=180)
    plt.close(fig)

    # 4. Gradient norm.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [item["gradient_norm"] for item in history])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean batch gradient norm")
    ax.set_title("Gradient norm during training")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "04_gradient_norm.png", dpi=180)
    plt.close(fig)

    # 5. Actual-vs-predicted scatter snapshots during training.
    best_val_eval = evaluate(model, val_loader, device, y_mean, y_std)
    snapshots[best_epoch] = (
        best_val_eval["targets"].copy(),
        best_val_eval["predictions"].copy(),
    )
    for snapshot_epoch in sorted(snapshots):
        snap_targets, snap_predictions = snapshots[snapshot_epoch]
        save_scatter(snap_targets, snap_predictions, plot_dir,
                     f"validation_epoch_{snapshot_epoch:03d}")

    # Final held-out test plots and residuals.
    save_scatter(test_target, test_pred, plot_dir, "final_test")
    save_residual_plots(test_target, test_pred, plot_dir)

    metrics = {
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "validation_mse_normalized": float(best_val),
        "test_rmse_rad_s2": {
            name: float(value) for name, value in zip(TARGET_NAMES, test_rmse)
        },
        "test_mae_rad_s2": {
            name: float(value) for name, value in zip(TARGET_NAMES, test_mae)
        },
        "test_samples": int(test_mask.sum()),
        "test_trajectories": int(len(np.unique(trajectory_ids[test_mask]))),
        "split": split_description,
        "checkpoint": str(output_path),
        "history_csv": str(history_csv),
        "plot_directory": str(plot_dir),
    }
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print("\nHeld-out test results (physical units):")
    for name, rmse_value, mae_value in zip(TARGET_NAMES, test_rmse, test_mae):
        print(f"  {name}: RMSE={rmse_value:.8g} rad/s² | MAE={mae_value:.8g} rad/s²")
    print(f"Best epoch: {best_epoch}")
    print(f"Checkpoint: {output_path}")
    print(f"Metrics:    {metrics_path}")
    print(f"History:    {history_csv}")
    print(f"Plots:      {plot_dir}")


if __name__ == "__main__":
    main()
