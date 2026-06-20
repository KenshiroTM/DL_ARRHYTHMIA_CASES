import torch

from database_processing import process_vitaldb
from model_training import evaluate_autoencoder, train_autoencoder, setup_device, export_hyperparams
from model_training.autoencoder import split_data_encoder
from models.architectures import MultimodalAutoencoder, build_multimodal_vae
from models.config import ModelConfig


def run_anomaly_pipeline(config: ModelConfig):
    c = config.copy()
    data = process_vitaldb()
    print(f"[MAIN] Załadowano {len(data)} rekordów z VitalDB")
    print(f"[MAIN] Rozkład klas:\n{data['scp_codes'].value_counts()}")

    print(f"\n{'='*50}")
    print(f"[anomaly] Multimodal Anomaly Detection — sygnały: {', '.join(c.signal_cols)}")
    print(f"{'='*50}")
    device = setup_device()
    train_df, val_df, test_df = split_data_encoder(data, train_size=0.8, val_size=0.1, seed=42)
    # Inicjalizacja autoenkodera dla 2 kanałów (PPG + EKG)
    model = build_multimodal_vae(kl_weight=0.001) # 1 bo każdy branch przyjmuje jedno
    model = model.to(device)

    if not c.save_path.exists():
        train_autoencoder(model, device, train_df, val_df, c)
    else:
        print(f"[anomaly] Wagi istnieją — pomijam trening: {c.save_path}")
        model.load_state_dict(torch.load(c.save_path, map_location=device, weights_only=False))

    export_hyperparams(config)

    evaluate_autoencoder(model, device, val_df, test_df, c, percentile_threshold=95)