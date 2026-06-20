from config import DatasetType
from models.config.ModelConfig import ModelConfig, WeightMethod, OptimizerType, SchedulerType
from runs.run_anomaly_pipeline import run_anomaly_pipeline
from runs.run_classification_pipeline import run_classification_pipeline

use_multimodal = True # czy wybrać multimodalne czy nie
def main():
    # Konfiguracje modeli:
    if not use_multimodal:
        experiments = [
            (ModelConfig(experiment_name="customECG", epochs=50, patience=5, weight_decay=1e-4,
                         optimizer=OptimizerType.ADAM, class_weight_method=WeightMethod.NONE,
                         augment=True), False, False),
            (ModelConfig(experiment_name="ECGFounde_with_weights", epochs=50, patience=5, weight_decay=1e-4,
                         optimizer=OptimizerType.ADAM, class_weight_method=WeightMethod.NONE,
                         augment=True), False, False),
        ]
        for config, use_ecg_architecture, use_ecg_weights in experiments:
            print(f"\n{'=' * 50}")
            print(f"[MAIN] Eksperyment: {config.experiment_name}")
            print(f"       Architektura: {'ECG' if use_ecg_architecture else 'Custom'}")
            print(f"       Wagi: {'pretrained' if use_ecg_weights else 'własne/brak'}")
            print(f"{'=' * 50}")

            run_classification_pipeline(config, use_ecg_architecture, use_ecg_weights, DatasetType.BINARY)
            run_classification_pipeline(config, use_ecg_architecture, use_ecg_weights, DatasetType.ARRHYTHMIA)
        print(f"\n[MAIN] Wszystkie eksperymenty zakończone!")

    # ── ANOMALY DETECTION (VitalDB) ───────────────────────────
    if use_multimodal: # TODO: na za tydzień rozrysowanie architektury + dane. Może spróbować na osobnym sygnale trenować i sprawdzić.
        experiments =[
            ModelConfig(
                experiment_name="anomaly_vae_small_batch",
                signal_cols=["ppg_signal", "ecg_signal"],
                batch_size=16,
                epochs=50,
                patience=5,
                lr=1e-4,
                weight_decay=1e-4,
            ),
        ]

        for config in experiments:
            print(f"\n{'=' * 50}")
            print(f"[MAIN] Eksperyment: {config.experiment_name}")
            print(f"{'=' * 50}")
            run_anomaly_pipeline(config)
        print(f"\n[MAIN] Anomaly detection zakończone!")

if __name__ == "__main__":
    main()