import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn, autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.config.ModelConfig import ModelConfig, SchedulerType, OptimizerType
from models.data_loaders.Dataset_loader import DatasetLoader


def prepare_dataframe(df: pd.DataFrame, signal_cols: list[str]) -> pd.DataFrame:
    df_processed = df.copy()
    for col in signal_cols:
        df_processed[col] = df_processed[col].apply(_prepare_and_normalize_signal)
    return df_processed


def _prepare_and_normalize_signal(signal: np.ndarray) -> np.ndarray:
    if not isinstance(signal, np.ndarray):
        return np.zeros((1, 5000), dtype=np.float32)

    if signal.ndim == 1:
        signal = signal[:5000] if len(signal) >= 5000 else np.pad(signal, (0, 5000 - len(signal)))
        signal = signal.reshape(1, 5000)
    elif signal.ndim == 2:
        signal = signal[:, :5000]
        if signal.shape[1] < 5000:
            signal = np.pad(signal, ((0, 0), (0, 5000 - signal.shape[1])))

    signal = signal.astype(np.float32)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    for i in range(signal.shape[0]):
        std = signal[i].std()
        if std > 1e-8:
            signal[i] = (signal[i] - signal[i].mean()) / std

    return signal


def train_autoencoder(
    model: nn.Module,
    device: torch.device,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_config: ModelConfig,
) -> tuple:
    c = model_config

    use_amp = c.amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)

    # Tylko NORM do treningu
    norm_df = train_df[train_df[c.label_col] == "NORM"].copy()
    norm_val_df = val_df[val_df[c.label_col] == "NORM"].copy()
    print(f"[anomaly] Trening na {len(norm_df)} sygnałach NORM (z {len(train_df)} łącznie)")
    print(f"[anomaly] Walidacja na {len(norm_val_df)} sygnałach NORM (z {len(val_df)} łącznie)")

    norm_df_proc = prepare_dataframe(norm_df, c.signal_cols)
    norm_val_proc = prepare_dataframe(norm_val_df, c.signal_cols)

    dataset = DatasetLoader(norm_df_proc, class_map={"NORM": 0}, signal_cols=c.signal_cols,
                            label_col=c.label_col, augment=False)
    val_dataset = DatasetLoader(norm_val_proc, class_map={"NORM": 0}, signal_cols=c.signal_cols,
                                label_col=c.label_col, augment=False)

    train_loader = DataLoader(dataset, batch_size=c.batch_size, shuffle=True, drop_last=True,
                              num_workers=c.num_workers, pin_memory=c.pin_memory,
                              persistent_workers=c.persistent_workers and c.num_workers > 0)

    val_loader = DataLoader(val_dataset, batch_size=c.batch_size, shuffle=False,
                            num_workers=c.num_workers, pin_memory=c.pin_memory,
                            persistent_workers=c.persistent_workers and c.num_workers > 0)

    if c.compile_model and hasattr(torch, 'compile') and device.type == 'cuda' and not (hasattr(torch.version, 'hip') and torch.version.hip):
        model = torch.compile(model, mode="reduce-overhead")

    if c.optimizer == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    elif c.optimizer == OptimizerType.ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    elif c.optimizer == OptimizerType.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=c.lr, momentum=c.momentum, weight_decay=c.weight_decay)
    else:
        raise ValueError(f"Nieznany optimizer: {c.optimizer}")

    scheduler = None
    if c.scheduler == SchedulerType.COSINE:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs)
    elif c.scheduler == SchedulerType.STEP:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=c.step_size, gamma=c.gamma)
    elif c.scheduler == SchedulerType.PLATEAU:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=c.sched_patience)

    criterion = nn.MSELoss()
    history = {"epoch": [], "train_loss": [], "val_loss": [], "lr": []}

    best_loss = float('inf')
    patience_counter = 0

    # Sprawdź czy to VAE
    is_vae = hasattr(model, 'loss_function')

    model.train()
    for epoch in range(c.epochs):
        total_loss = 0.0

        for sample, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{c.epochs}"):
            nb = c.pin_memory and device.type == 'cuda'
            ppg = sample["ppg_signal"].to(device, non_blocking=nb)
            ecg = sample["ecg_signal"].to(device, non_blocking=nb)

            optimizer.zero_grad()

            with autocast(device_type='cuda', enabled=use_amp):
                if is_vae:
                    recon_ppg, recon_ecg, mu_ppg, logvar_ppg, mu_ecg, logvar_ecg = model(ppg, ecg)
                    loss, _, _ = model.loss_function(
                        recon_ppg, ppg, recon_ecg, ecg,
                        mu_ppg, logvar_ppg, mu_ecg, logvar_ecg
                    )
                else:
                    recon_ppg, recon_ecg = model(ppg, ecg)
                    loss = criterion(recon_ppg, ppg) + criterion(recon_ecg, ecg)

            if torch.isnan(loss):
                print(f"[anomaly] UWAGA: NaN w loss — pomijam batch")
                continue

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

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else float("nan")
        current_lr = optimizer.param_groups[0]["lr"]

        # Val loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sample, _ in val_loader:
                nb = c.pin_memory and device.type == 'cuda'
                ppg = sample["ppg_signal"].to(device, non_blocking=nb)
                ecg = sample["ecg_signal"].to(device, non_blocking=nb)
                with autocast(device_type='cuda', enabled=use_amp):
                    if is_vae:
                        recon_ppg, recon_ecg, mu_ppg, logvar_ppg, mu_ecg, logvar_ecg = model(ppg, ecg)
                        v_loss, _, _ = model.loss_function(
                            recon_ppg, ppg, recon_ecg, ecg,
                            mu_ppg, logvar_ppg, mu_ecg, logvar_ecg
                        )
                    else:
                        recon_ppg, recon_ecg = model(ppg, ecg)
                        v_loss = criterion(recon_ppg, ppg) + criterion(recon_ecg, ecg)
                val_loss += v_loss.item()
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float("nan")
        model.train()

        # Early stopping
        if avg_val_loss < best_loss - c.min_delta:
            best_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"Epoch {epoch + 1}/{c.epochs} — train: {avg_train_loss:.6f} | val: {avg_val_loss:.6f} | lr: {current_lr:.6f} | patience: {patience_counter}/{c.patience}")

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
            print(f"[anomaly] Early stopping po {epoch + 1} epokach.")
            break

    if c.save_path is not None:
        print("Zapisywanie wag modelu...")
        state_dict = model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict()
        torch.save(state_dict, c.save_path)

    output_dir = Path("results") / c.experiment_name
    _plot_training_history(history, output_dir, c.signal_cols)

    return model, device


def _plot_training_history(history: dict, output_dir: Path, signal_cols: list):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Historia trenowania autoenkodera — {', '.join(signal_cols)}", fontsize=14, fontweight="bold")

    axes[0].plot(history["epoch"], history["train_loss"], color="#2E75B6", linewidth=2, marker="o", markersize=3, label="Train")
    axes[0].plot(history["epoch"], history["val_loss"], color="#C0504D", linewidth=2, marker="s", markersize=3, label="Val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].set_xlabel("Epoka")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["epoch"], history["lr"], color="#7030A0", linewidth=2, marker="d", markersize=3)
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Epoka")
    axes[1].set_ylabel("LR")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    plt.tight_layout()
    fig.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    print(f"[anomaly] Historia zapisana: {output_dir}")