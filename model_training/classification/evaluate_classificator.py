import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from models.config.ModelConfig import ModelConfig
from models.data_loaders.Dataset_loader import DatasetLoader

def evaluate_classificator(
    model,
    device,
    df: pd.DataFrame,
    class_map: dict,
    model_config: ModelConfig,
    split_name: str = "Val",
    output_dir: Path = Path("results"),
) -> dict:

    c = model_config

    dataset = DatasetLoader(df, class_map, c.signal_cols, c.feature_cols)
    loader = DataLoader(dataset, batch_size=c.batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for sample, labels in loader:
            labels = labels.to(device)

            inputs = {k: v.to(device) for k, v in sample.items()}
            outputs = model(**inputs)

            total_loss += criterion(outputs, labels).item()
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    acc = accuracy_score(all_labels, all_preds)

    report = classification_report(
        all_labels, all_preds,
        target_names=list(class_map.keys()),
        output_dict=True
    )

    cm = confusion_matrix(all_labels, all_preds)

    # --- EKSPORT WSZYSTKICH PARAMETRÓW I WYNIKÓW ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wszystkie pola z ModelConfig
    config_dict = {f.name: getattr(c, f.name) for f in ModelConfig.__dataclass_fields__.values()}
    # Zamień enumy na stringi
    for k, v in config_dict.items():
        if isinstance(v, Path):
            config_dict[k] = str(v)
        elif hasattr(v, "value"):
            config_dict[k] = v.value

    # --- EKSPORT WYNIKÓW ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Wyniki per klasa
    per_class = {}
    for cls_name, metrics in report.items():
        if cls_name in ["accuracy", "macro avg", "weighted avg"]:
            continue
        per_class[cls_name] = {
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1-score": round(metrics["f1-score"], 4),
            "support": int(metrics["support"]),
        }

    json_path = output_dir / f"{split_name}_report.json"

    json_data = {
        "eval_info": {
            "split": split_name,
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "metrics": {
            "loss": round(avg_loss, 4),
            "accuracy": round(acc, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_micro": round(f1_micro, 4),
        },
        "per_class": per_class,
        "classification_report": report,
        "model_config": config_dict,
        "confusion_matrix": cm.tolist(),
        "predictions_summary": {
            "total_samples": len(all_labels),
            "correct_predictions": int(sum(1 for p, l in zip(all_preds, all_labels) if p == l)),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Predykcje CSV
    pd.DataFrame({"preds": all_preds, "labels": all_labels}).to_csv(
        output_dir / f"{split_name}_predictions.csv", index=False
    )

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(class_map.keys()),
                yticklabels=list(class_map.keys()))
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Rzeczywistość")
    ax.set_title(f"Confusion Matrix — {split_name}")
    plt.tight_layout()
    fig.savefig(output_dir / f"{split_name}_confusion_matrix.png")
    plt.close()

    # Metryki per klasa (bar plot)
    metrics_df = pd.DataFrame(report).T
    metrics_df = metrics_df.drop(["accuracy", "macro avg", "weighted avg"], errors="ignore")

    fig, ax = plt.subplots(figsize=(12, 6))
    metrics_df[["precision", "recall", "f1-score"]].plot(kind="bar", ax=ax)
    ax.set_title(f"Metryki per klasa — {split_name}")
    ax.set_xlabel("Klasa")
    ax.set_ylabel("Wartość")
    ax.legend(loc="lower right")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / f"{split_name}_metrics.png")
    plt.close()

    print(f"[evaluate] {split_name} — Loss: {avg_loss:.4f}, Acc: {acc:.4f}, F1: {f1_weighted:.4f}")

    return {
        "loss": avg_loss,
        "accuracy": acc,
        "f1": f1_weighted,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "preds": all_preds,
        "labels": all_labels,
        "report": report,
    }