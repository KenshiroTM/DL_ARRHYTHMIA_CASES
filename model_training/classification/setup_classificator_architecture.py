from pathlib import Path

import torch

from models.architectures.custom_network import build_custom_ecg_net
from models.architectures.ecg_net1d import Net1D


def setup_classificator_architecture(device: torch.device, model_path: Path | None, n_classes: int, in_channels: int, use_ecg_architecture: bool = True):
    if use_ecg_architecture:
        model = Net1D(
            in_channels=in_channels,
            base_filters=64,
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=n_classes
        )
    else:
        print("[setup] Wczytano classification customowy.") # porównuje wagi modelu z setupowanymi
        model = build_custom_ecg_net(n_classes=n_classes, in_channels=in_channels)
    if model_path is not None and model_path.exists():
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        model_state = model.state_dict()
        filtered = {}

        for k, v in checkpoint.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v

        model.load_state_dict(filtered, strict=False)
        print(f"[setup] Wczytano {len(filtered)}/{len(checkpoint)} wag z: {model_path}")
    else:
        print(f"[setup] Brak wag pre-trenowanych — classification zainicjalizowany losowo.")

    model.to(device)
    model.eval()
    return model, device