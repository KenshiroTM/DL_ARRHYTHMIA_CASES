import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from models.config.ModelConfig import ModelConfig, OptimizerType, SchedulerType, WeightMethod
from models.data_loaders.Dataset_loader import DatasetLoader


def train_classificator(
        model,
        device,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        class_map: dict,
        model_config: ModelConfig
) -> nn.Module:
    c = model_config

    # AMP z configu
    use_amp = c.amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # Augmentacja w DatasetLoader
    train_dataset = DatasetLoader(
        train_df,
        class_map,
        c.signal_cols,
        c.feature_cols,
        label_col = c.label_col,
        augment=c.augment,
        noise_std=c.noise_std,
        scale_low=c.scale_low,
        scale_high=c.scale_high,
        time_shift_pct=c.time_shift_pct
    )

    val_dataset = DatasetLoader(val_df, class_map, c.signal_cols, c.feature_cols, label_col=c.label_col)

    # Class weights do lossa i ewentualnie sampler
    class_weights = None
    if c.class_weight_method is not None and c.class_weight_method != WeightMethod.NONE:
        class_weights = compute_class_weights(train_df, c.label_col, class_map, c.class_weight_method)
        class_weights = class_weights.to(device)

    # WeightedRandomSampler jeśli class weights
    sampler = None
    if class_weights is not None:
        sample_weights = get_sample_weights(train_df, c.label_col, class_weights, class_map)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DataLoader z configu
    train_loader = DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=c.num_workers,
        pin_memory=c.pin_memory,
        persistent_workers=c.persistent_workers and c.num_workers > 0,
    )

    val_loader = DataLoader(val_dataset, batch_size=c.batch_size, shuffle=False,
                            num_workers=c.num_workers, pin_memory=c.pin_memory)
    # Model compile z configu
    if c.compile_model and hasattr(torch, 'compile') and device.type == 'cuda' and not (
            hasattr(torch.version, 'hip') and torch.version.hip):
        model = torch.compile(model, mode="reduce-overhead")

    # Optimizer
    if c.optimizer == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    elif c.optimizer == OptimizerType.ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    elif c.optimizer == OptimizerType.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay)
    else:
        raise ValueError(f"Nieznany optimizer: {c.optimizer}")

    # Scheduler
    scheduler = None
    if c.scheduler == SchedulerType.COSINE:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
    elif c.scheduler == SchedulerType.STEP:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c.step_size, gamma=c.gamma)
    elif c.scheduler == SchedulerType.PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=c.sched_patience)

    # Loss — CrossEntropy z class weights LUB Focal Loss
    if c.focal_gamma > 0:
        criterion = FocalLoss(gamma=c.focal_gamma, alpha=c.focal_alpha, weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    # Early stopping
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    for epoch in range(c.epochs):
        train_loss = 0.0

        for sample, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{c.epochs}"):
            nb = c.pin_memory and device.type == 'cuda'
            labels = labels.to(device, non_blocking=nb)
            inputs = {
                ("x" if k == "signal" else k): v.to(device, non_blocking=nb)
                for k, v in sample.items()
            }

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=use_amp):
                outputs = model(**inputs)
                loss = criterion(outputs, labels)

            # Gradient clipping
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sample, labels in val_loader:
                labels = labels.to(device)
                inputs = {("x" if k == "signal" else k): v.to(device) for k, v in sample.items()}
                outputs = model(**inputs)
                val_loss += criterion(outputs, labels).item()
        avg_val_loss = val_loss / len(val_loader)
        model.train()

        # Early stopping check
        if avg_val_loss < best_loss - c.min_delta:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        print(f"Epoch {epoch + 1}/{c.epochs} — train_loss: {avg_train_loss:.4f} | val_loss: {avg_val_loss:.4f} | lr: {current_lr:.6f} | patience: {patience_counter}/{c.patience}")

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["lr"].append(current_lr)

        if scheduler is not None:
            if c.scheduler == SchedulerType.PLATEAU:
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        if patience_counter >= c.patience:
            print(f"Early stopping po {epoch + 1} epokach.")
            break

    # Zapis wag
    if c.save_path is not None:
        print("Zapisywanie wag modelu...")
        state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state_dict, c.save_path)

    _plot_training_history(history, c)
    return model, device

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss

        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[targets]
                loss = alpha_t * loss
            else:
                # alpha to float — globalny mnożnik
                loss = self.alpha * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

def compute_class_weights(df, label_col, class_map, method):
    """Liczy wagi klas na podstawie częstości w dataset."""
    counts = df[label_col].value_counts().sort_index()
    num_classes = len(class_map)

    full_counts = np.zeros(num_classes)
    for label, count in counts.items():
        if label in class_map:
            idx = class_map[label]
            full_counts[idx] = count

    full_counts = np.where(full_counts == 0, 1, full_counts)

    if method == WeightMethod.INVERSE:
        weights = 1.0 / full_counts
    elif method == WeightMethod.SQRT:
        weights = 1.0 / np.sqrt(full_counts)
    elif method == WeightMethod.EFFECTIVE:
        beta = 0.9999
        weights = (1.0 - beta) / (1.0 - beta ** full_counts)
    else:
        weights = np.ones(num_classes)

    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def get_sample_weights(df, label_col, class_weights, class_map):
    indices = df[label_col].map(class_map).values.astype(int)
    return class_weights.cpu()[indices].numpy()


def _plot_training_history(history: dict, c: ModelConfig):
    output_dir = Path("results") / c.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Historia trenowania — {c.experiment_name}", fontsize=14, fontweight="bold")

    # Train loss - niebieski, Val loss - czerwony
    axes[0].plot(epochs, history["train_loss"], color="#2E75B6", linewidth=2, marker="o", markersize=4,
                 label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], color="#C0504D", linewidth=2, marker="s", markersize=4,
                 label="Val Loss")
    axes[0].set_title("Loss per epoka")
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["lr"], color="#70AD47", linewidth=2, marker="^", markersize=4, label="Learning Rate")
    axes[1].set_title("Learning Rate per epoka")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("LR")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    out_path = output_dir / f"training_history_{c.model_type}.png"
    fig.savefig(out_path, dpi=150)
    plt.close()
    print(f"[train] Wykres historii zapisany: {out_path}")

    pd.DataFrame(history).to_csv(output_dir / f"training_history_{c.model_type}.csv", index=False)
    print(f"[train] Historia epok zapisana: {output_dir / f'training_history_{c.model_type}.csv'}")