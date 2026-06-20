import subprocess

import torch


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name()
        try:
            subprocess.run(["rocm-smi"], capture_output=True, check=True)
            backend = "ROCm (AMD)"
        except (FileNotFoundError, subprocess.CalledProcessError):
            backend = "CUDA (NVIDIA)" if "NVIDIA" in device_name else f"CPU (fallback — brak wsparcia dla: {device_name})"
            device = torch.device("cpu") if "NVIDIA" not in device_name else device
    else:
        device = torch.device("cpu")
        device_name = "CPU"
        backend = "CPU"
    print(f"[setup] {backend} — {device_name}")
    return device