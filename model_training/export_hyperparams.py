from pathlib import Path
from config import results_folder
from models.config import ModelConfig

# Pola które nigdy nie są hiperparametrami
_SKIP_FIELDS = {"save_path", "model_type", "num_workers", "persistent_workers", "pin_memory"}

# Grupowanie pól dla czytelności
_GROUPS = {
    "Trening": ["epochs", "batch_size", "lr", "weight_decay", "optimizer", "scheduler"],
    "Early stopping": ["patience", "min_delta"],
    "Scheduler": ["step_size", "gamma", "sched_patience"],
    "Augmentacja": ["augment", "noise_std", "scale_low", "scale_high", "time_shift_pct"],
    "Ważenie klas": ["class_weight_method", "focal_gamma", "focal_alpha"],
    "Dane": ["signal_cols", "feature_cols", "label_col", "amp", "compile_model"],
}

# Pola zależne -- wypisywane tylko gdy warunek spełniony
_CONDITIONAL = {
    "momentum":       lambda c: c.optimizer.value == "sgd",
    "step_size":      lambda c: c.scheduler.value == "step",
    "gamma":          lambda c: c.scheduler.value == "step",
    "sched_patience": lambda c: c.scheduler.value == "plateau",
    "noise_std":      lambda c: c.augment,
    "scale_low":      lambda c: c.augment,
    "scale_high":     lambda c: c.augment,
    "time_shift_pct": lambda c: c.augment,
    "focal_alpha":    lambda c: c.focal_alpha is not None,
    "feature_cols":   lambda c: c.feature_cols is not None,
}


def _fmt(v):
    if isinstance(v, Path):
        return str(v)
    if hasattr(v, "value"):
        return v.value
    return v


def _should_skip(name: str, value, c) -> bool:
    """Zwraca True jeśli pole powinno być pominięte."""
    if name in _SKIP_FIELDS:
        return True
    if value is None:
        return True
    if isinstance(value, bool) and value is False:
        return True
    if name in _CONDITIONAL and not _CONDITIONAL[name](c):
        return True
    return False


def export_hyperparams(c: ModelConfig) -> None:
    """
    Eksportuje hiperparametry modelu do pliku hyperparameters.txt
    w results_folder / experiment_name. Wywołać raz per eksperyment przed ewaluacją.
    """
    output_dir = results_folder / c.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / "hyperparameters.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  HIPERPARAMETRY — {c.experiment_name}\n")
        f.write("=" * 60 + "\n\n")

        for group, field_names in _GROUPS.items():
            # sprawdź czy jakakolwiek linia w grupie się wypisze
            visible = [
                name for name in field_names
                if not _should_skip(name, getattr(c, name), c)
            ]
            if not visible:
                continue  # pomiń całą grupę jeśli pusta

            f.write(f"{group}:\n")
            for name in visible:
                val = _fmt(getattr(c, name))
                f.write(f"  {name}: {val}\n")
            f.write("\n")

    print(f"[hyperparams] Zapisano: {txt_path}")