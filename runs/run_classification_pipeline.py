import copy

from config import *
from database_processing import process_arrhythmia, process_ptbxl
from model_training import setup_classificator_architecture, setup_device
from model_training.classification.evaluate_classificator import evaluate_classificator
from model_training.classification.train_classificator import train_classificator
from model_training import export_hyperparams
from model_training.classification.split_data_classificator import split_data_classificator
from models.config.ModelConfig import ModelConfig

def run_classification_pipeline(current_model: ModelConfig, use_ecg_architecture: bool, use_ecg_weights: bool, dataset_type: DatasetType):
    # === WYBÓR WAG ===
    current_model = copy.deepcopy(current_model) # skopiować bo wtedy zmieni obiekt inferencją
    class_map = None
    data = None
    current_model.save_path = current_model.save_path.with_stem(f"{current_model.save_path.stem}_{dataset_type.value}")
    current_model.model_type = dataset_type.value
    # dodanie _binary lub _arrhythmia do przyrostka

    if use_ecg_weights and not current_model.save_path.exists(): # jeżeli używać ma wag ecgfoundera i nie ma pathu z pretrenowanego ecg to wtedy trening
        weights_to_load = ECG_PRETRAINED_WEIGHTS
        print(f"[run_config] Używam wag pretrainowanych: {weights_to_load}")
    else:
        weights_to_load = current_model.save_path if current_model.save_path.exists() else None
        print(f"[run_config] Używam własnych wag: {weights_to_load}")

    if dataset_type is DatasetType.ARRHYTHMIA:
        data = process_arrhythmia()
        class_map = arr_map
    elif dataset_type is DatasetType.BINARY:
        data = process_ptbxl()
        class_map = bin_map

    train_df, val_df, test_df = split_data_classificator(data, train_size, val_size, seed)

    # === MODEL ===
    device = setup_device()
    model, device = setup_classificator_architecture(
        device,
        model_path=weights_to_load,
        n_classes=len(class_map),
        in_channels=n_leads,
        use_ecg_architecture=use_ecg_architecture,
    )

    # === TRENING ===
    if not current_model.save_path.exists():
        print(f"[run_config] Brak wag w {current_model.save_path} — rozpoczynam trening...")
        model, device = train_classificator(model, device, train_df,  val_df, class_map, current_model)
    else:
        print(f"[run_config] Wagi istnieją w {current_model.save_path} — pomijam trening.")

    # === EWALUACJA ===
    # Wyniki w podfolderze zależnym od nazwy eksperymentu
    output_base = results_folder / current_model.experiment_name

    export_hyperparams(current_model) # tu eksport hiperparametrów do osobnego txt

    evaluate_classificator(
        model, device, val_df, class_map, current_model,
        split_name="Val",
        output_dir=output_base / f"{dataset_type.value}_validation"
    )

    if test_df is not None:
        evaluate_classificator(
            model, device, test_df, class_map, current_model,
            split_name="Test",
            output_dir=output_base / f"{dataset_type.value}_test"
        )
    else:
        print("[run_config] Brak zbioru testowego")