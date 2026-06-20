from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader

from model_training.autoencoder.train_autoencoder import prepare_dataframe
from models.config.ModelConfig import ModelConfig
from models.data_loaders.Dataset_loader import DatasetLoader


def evaluate_autoencoder(
        model: nn.Module,
        device: torch.device,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_config: ModelConfig,
        percentile_threshold: float = 95.0,
) -> dict:
    c = model_config
    output_base = Path("results") / c.experiment_name
    output_base.mkdir(parents=True, exist_ok=True)
    signal_cols = c.signal_cols

    model.eval()
    class_map = {"NORM": 0}

    # Sprawdź czy to VAE
    is_vae = hasattr(model, 'loss_function')

    def get_errors_and_labels(df):
        df_proc = prepare_dataframe(df, signal_cols)
        dataset = DatasetLoader(df_proc, class_map=class_map, signal_cols=signal_cols, label_col=c.label_col)
        loader = DataLoader(
            dataset,
            batch_size=c.batch_size,
            shuffle=False,
            num_workers=c.num_workers,
            pin_memory=c.pin_memory,
            persistent_workers=c.persistent_workers and c.num_workers > 0,
        )

        errors_mse = []
        errors_neural_rhythm = []
        errors_classical_rhythm = []
        errors_combined = []

        with torch.no_grad():
            for sample, _ in loader:
                nb = c.pin_memory and device.type == 'cuda'
                ppg = sample["ppg_signal"].to(device, non_blocking=nb)
                ecg = sample["ecg_signal"].to(device, non_blocking=nb)

                # MSE z autoenkodera
                if is_vae:
                    recon_ppg, recon_ecg, mu_ppg, logvar_ppg, mu_ecg, logvar_ecg = model(ppg, ecg)
                else:
                    recon_ppg, recon_ecg = model(ppg, ecg)

                mse_ppg = ((recon_ppg - ppg) ** 2).mean(dim=(1, 2))
                mse_ecg = ((recon_ecg - ecg) ** 2).mean(dim=(1, 2))
                mse_total = mse_ppg + mse_ecg

                # Neural Rhythm — std R-R intervals z modelu
                _, rr_intervals = model.rhythm_enc(ecg)
                neural_rhythm = rr_intervals.std(dim=1)

                # Classical Rhythm
                classical_rhythm = rr_regularity_error(ecg)

                # Combined
                combined = mse_total.cpu().numpy() + 0.5 * neural_rhythm.cpu().numpy()

                errors_mse.extend(mse_total.cpu().numpy())
                errors_neural_rhythm.extend(neural_rhythm.cpu().numpy())
                errors_classical_rhythm.extend(classical_rhythm.cpu().numpy())
                errors_combined.extend(combined)

        labels = np.where(df[c.label_col] == "NORM", 0, 1)

        errors_mse = np.nan_to_num(np.array(errors_mse), nan=0.0)
        errors_neural_rhythm = np.nan_to_num(np.array(errors_neural_rhythm), nan=0.0)
        errors_classical_rhythm = np.nan_to_num(np.array(errors_classical_rhythm), nan=0.0)
        errors_combined = np.nan_to_num(np.array(errors_combined), nan=0.0)

        return errors_mse, errors_neural_rhythm, errors_classical_rhythm, errors_combined, labels

    print("[anomaly] Obliczanie błędów rekonstrukcji (val)...")
    val_mse, val_neural, val_classical, val_comb, val_labels = get_errors_and_labels(val_df)

    print("[anomaly] Obliczanie błędów rekonstrukcji (test)...")
    test_mse, test_neural, test_classical, test_comb, test_labels = get_errors_and_labels(test_df)

    # PRINTY PORÓWNAWCZE
    print("\n" + "=" * 70)
    print("PORÓWNANIE BŁĘDÓW: NORM vs ANOMALIA")
    print("=" * 70)

    for split_name, mse, neural, classical, comb, labels in [
        ("VAL", val_mse, val_neural, val_classical, val_comb, val_labels),
        ("TEST", test_mse, test_neural, test_classical, test_comb, test_labels),
    ]:
        norm_mask = labels == 0
        anom_mask = labels == 1

        print(f"\n--- {split_name} ---")
        print(f"  MSE              — NORM: {mse[norm_mask].mean():.6f} ± {mse[norm_mask].std():.6f}")
        if anom_mask.sum() > 0:
            print(f"  MSE              — ANOM: {mse[anom_mask].mean():.6f} ± {mse[anom_mask].std():.6f}")
        else:
            print(f"  MSE              — ANOM: brak próbek")

        print(f"  NEURAL RHYTHM    — NORM: {neural[norm_mask].mean():.6f} ± {neural[norm_mask].std():.6f}")
        if anom_mask.sum() > 0:
            print(f"  NEURAL RHYTHM    — ANOM: {neural[anom_mask].mean():.6f} ± {neural[anom_mask].std():.6f}")
        else:
            print(f"  NEURAL RHYTHM    — ANOM: brak próbek")

        print(f"  CLASSICAL RHYTHM — NORM: {classical[norm_mask].mean():.6f} ± {classical[norm_mask].std():.6f}")
        if anom_mask.sum() > 0:
            print(f"  CLASSICAL RHYTHM — ANOM: {classical[anom_mask].mean():.6f} ± {classical[anom_mask].std():.6f}")
        else:
            print(f"  CLASSICAL RHYTHM — ANOM: brak próbek")

        print(f"  COMB             — NORM: {comb[norm_mask].mean():.6f} ± {comb[norm_mask].std():.6f}")
        if anom_mask.sum() > 0:
            print(f"  COMB             — ANOM: {comb[anom_mask].mean():.6f} ± {comb[anom_mask].std():.6f}")
        else:
            print(f"  COMB             — ANOM: brak próbek")
    print("=" * 70)

    # Próg z błędów COMBINED na zbiorze walidacyjnym
    norm_errors_val = val_comb[val_labels == 0]
    threshold = np.percentile(norm_errors_val, percentile_threshold)
    print(f"\n[anomaly] Próg anomalii (percentyl {percentile_threshold}): {threshold:.6f}")

    val_preds = (val_comb > threshold).astype(int)
    test_preds = (test_comb > threshold).astype(int)

    results = {}
    for split_name, mse, neural, classical, comb, labels, preds in [
        ("Val", val_mse, val_neural, val_classical, val_comb, val_labels, val_preds),
        ("Test", test_mse, test_neural, test_classical, test_comb, test_labels, test_preds),
    ]:
        output_dir = output_base / split_name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        has_both_classes = len(np.unique(labels)) > 1

        auroc_mse = roc_auc_score(labels, mse) if has_both_classes else float("nan")
        auroc_neural = roc_auc_score(labels, neural) if has_both_classes else float("nan")
        auroc_classical = roc_auc_score(labels, classical) if has_both_classes else float("nan")
        auroc_comb = roc_auc_score(labels, comb) if has_both_classes else float("nan")

        f1 = f1_score(labels, preds, zero_division=0)
        norm_count = (labels == 0).sum()
        anom_count = (labels == 1).sum()
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\n[anomaly] {split_name}:")
        print(f"  AUROC — MSE:              {auroc_mse:.4f}" if not np.isnan(
            auroc_mse) else "  AUROC — MSE:              brak")
        print(f"  AUROC — NEURAL RHYTHM:    {auroc_neural:.4f}" if not np.isnan(
            auroc_neural) else "  AUROC — NEURAL RHYTHM:    brak")
        print(f"  AUROC — CLASSICAL RHYTHM: {auroc_classical:.4f}" if not np.isnan(
            auroc_classical) else "  AUROC — CLASSICAL RHYTHM: brak")
        print(f"  AUROC — COMB:             {auroc_comb:.4f}" if not np.isnan(
            auroc_comb) else "  AUROC — COMB:             brak")
        print(f"  F1 (COMB):                {f1:.4f}")
        print(f"  Precision:                {precision:.4f}")
        print(f"  Recall:                   {recall:.4f}")
        print(f"  NORM: {norm_count} | ANOMALIA: {anom_count}")

        # Przygotowanie config_dict
        config_dict = {f.name: getattr(c, f.name) for f in ModelConfig.__dataclass_fields__.values()}
        for k, v in config_dict.items():
            if isinstance(v, Path):
                config_dict[k] = str(v)
            elif hasattr(v, "value"):
                config_dict[k] = v.value

        json_path = output_dir / f"{split_name}_anomaly_report.json"

        json_data = {
            "eval_info": {
                "split": split_name,
                "experiment_name": c.experiment_name,
                "timestamp": pd.Timestamp.now().isoformat(),
            },
            "parameters": {
                "signals": signal_cols,
                "threshold": round(float(threshold), 6),
                "percentile_threshold": percentile_threshold,
                "model_config": config_dict,
            },
            "dataset_stats": {
                "norm_count": int(norm_count),
                "anomaly_count": int(anom_count),
                "total_samples": int(len(labels)),
            },
            "metrics": {
                "auroc_mse": round(float(auroc_mse), 4) if not np.isnan(auroc_mse) else None,
                "auroc_neural_rhythm": round(float(auroc_neural), 4) if not np.isnan(auroc_neural) else None,
                "auroc_classical_rhythm": round(float(auroc_classical), 4) if not np.isnan(auroc_classical) else None,
                "auroc_combined": round(float(auroc_comb), 4) if not np.isnan(auroc_comb) else None,
                "f1": round(float(f1), 4),
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
            },
            "confusion_matrix": {
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
            },
            "error_stats": {
                "mse_norm_mean": round(float(mse[labels == 0].mean()), 6) if norm_count > 0 else None,
                "mse_norm_std": round(float(mse[labels == 0].std()), 6) if norm_count > 0 else None,
                "mse_anom_mean": round(float(mse[labels == 1].mean()), 6) if anom_count > 0 else None,
                "mse_anom_std": round(float(mse[labels == 1].std()), 6) if anom_count > 0 else None,
                "neural_norm_mean": round(float(neural[labels == 0].mean()), 6) if norm_count > 0 else None,
                "neural_norm_std": round(float(neural[labels == 0].std()), 6) if norm_count > 0 else None,
                "neural_anom_mean": round(float(neural[labels == 1].mean()), 6) if anom_count > 0 else None,
                "neural_anom_std": round(float(neural[labels == 1].std()), 6) if anom_count > 0 else None,
                "classical_norm_mean": round(float(classical[labels == 0].mean()), 6) if norm_count > 0 else None,
                "classical_norm_std": round(float(classical[labels == 0].std()), 6) if norm_count > 0 else None,
                "classical_anom_mean": round(float(classical[labels == 1].mean()), 6) if anom_count > 0 else None,
                "classical_anom_std": round(float(classical[labels == 1].std()), 6) if anom_count > 0 else None,
                "comb_norm_mean": round(float(comb[labels == 0].mean()), 6) if norm_count > 0 else None,
                "comb_norm_std": round(float(comb[labels == 0].std()), 6) if norm_count > 0 else None,
                "comb_anom_mean": round(float(comb[labels == 1].mean()), 6) if anom_count > 0 else None,
                "comb_anom_std": round(float(comb[labels == 1].std()), 6) if anom_count > 0 else None,
            },
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        # === PORÓWNAWCZY WYKRES: NEURAL vs CLASSICAL ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Neural vs Classical Rhythm — {c.experiment_name} — {split_name}",
                     fontsize=14, fontweight="bold")

        # Histogramy porównawcze
        norm_mask = labels == 0
        anom_mask = labels == 1

        # Neural
        axes[0].hist(neural[norm_mask], bins=50, alpha=0.6, color="#2E75B6",
                     label=f"NORM (n={norm_mask.sum()})")
        if anom_mask.sum() > 0:
            axes[0].hist(neural[anom_mask], bins=50, alpha=0.6, color="#C0504D",
                         label=f"AFIB (n={anom_mask.sum()})")
        axes[0].set_title(f"Neural Rhythm (AUROC={auroc_neural:.3f})")
        axes[0].set_xlabel("Błąd")
        axes[0].set_ylabel("Liczba próbek")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Classical
        axes[1].hist(classical[norm_mask], bins=50, alpha=0.6, color="#2E75B6",
                     label=f"NORM (n={norm_mask.sum()})")
        if anom_mask.sum() > 0:
            axes[1].hist(classical[anom_mask], bins=50, alpha=0.6, color="#C0504D",
                         label=f"AFIB (n={anom_mask.sum()})")
        axes[1].set_title(f"Classical Rhythm (AUROC={auroc_classical:.3f})")
        axes[1].set_xlabel("Błąd")
        axes[1].set_ylabel("Liczba próbek")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(output_dir / f"{split_name}_rhythm_comparison.png", dpi=150)
        plt.close()

        # === PORÓWNAWCZY WYKRES ROC: NEURAL vs CLASSICAL ===
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        if has_both_classes:
            # Neural — czerwony
            fpr_n, tpr_n, _ = roc_curve(labels, neural)
            ax.plot(fpr_n, tpr_n, color="#C0504D", linewidth=2.5,
                    label=f"Neural Rhythm (AUROC = {auroc_neural:.4f})")

            # Classical — niebieski
            fpr_c, tpr_c, _ = roc_curve(labels, classical)
            ax.plot(fpr_c, tpr_c, color="#2E75B6", linewidth=2.5,
                    label=f"Classical Rhythm (AUROC = {auroc_classical:.4f})")

            # Baseline
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random (AUROC = 0.5000)")

            ax.set_title(f"ROC Comparison — {split_name}", fontsize=14, fontweight="bold")
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate", fontsize=12)
            ax.legend(loc="lower right", fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        else:
            ax.text(0.5, 0.5, "Za mało klas do ROC", ha="center", va="center",
                    transform=ax.transAxes, fontsize=14)

        plt.tight_layout()
        fig.savefig(output_dir / f"{split_name}_roc_comparison.png", dpi=150)
        plt.close()

        # === STANDARDOWY WYKRES COMBINED (jak wcześniej) ===
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Anomaly Detection — {c.experiment_name} — {split_name}", fontweight="bold")

        norm_err = comb[labels == 0]
        anom_err = comb[labels == 1]
        axes[0].hist(norm_err, bins=50, alpha=0.7, color="#2E75B6", label=f"NORM (n={len(norm_err)})")
        if len(anom_err) > 0:
            axes[0].hist(anom_err, bins=50, alpha=0.7, color="#C0504D", label=f"ANOMALIA (n={len(anom_err)})")
        axes[0].axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Próg={threshold:.4f}")
        axes[0].set_title("Rozkład błędu COMBINED")
        axes[0].set_xlabel("Błąd")
        axes[0].set_ylabel("Liczba próbek")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if has_both_classes:
            fpr, tpr, _ = roc_curve(labels, comb)
            axes[1].plot(fpr, tpr, color="#2E75B6", linewidth=2, label=f"AUROC = {auroc_comb:.4f}")
            axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[1].set_title("Krzywa ROC (COMBINED)")
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, "Za mało klas do ROC\n(brak anomalii w zbiorze)",
                         ha="center", va="center", transform=axes[1].transAxes)

        plt.tight_layout()
        fig.savefig(output_dir / f"{split_name}_anomaly_plot.png", dpi=150)
        plt.close()

        results[split_name] = {
            "auroc_mse": auroc_mse,
            "auroc_neural_rhythm": auroc_neural,
            "auroc_classical_rhythm": auroc_classical,
            "auroc_combined": auroc_comb,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "threshold": threshold,
        }

    return results

def rr_regularity_error(ecg_signal):
    """
    Klasyczny błąd regularności: std(RR) / mean(RR).
    AFIB = wysoki (nieregularny), NORM = niski (regularny).
    """
    from scipy.signal import find_peaks

    rr_intervals = []
    for b in range(ecg_signal.shape[0]):
        sig = ecg_signal[b, 0, :].cpu().numpy()
        peaks, _ = find_peaks(sig, height=np.std(sig) * 2, distance=500 // 3)
        if len(peaks) >= 2:
            rr = np.diff(peaks) / 500  # w sekundach
            rr_intervals.append(rr)
        else:
            rr_intervals.append(np.array([0.8]))

    errors = []
    for rr in rr_intervals:
        if len(rr) >= 2:
            cv = np.std(rr) / (np.mean(rr) + 1e-8)
            errors.append(cv)
        else:
            errors.append(0.0)
    return torch.tensor(errors, device=ecg_signal.device)