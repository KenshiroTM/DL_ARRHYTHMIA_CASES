import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class DatasetLoader(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_map: dict,
        signal_cols: list[str],
        feature_cols: list[str] = None,
        label_col: str = "scp_codes",
        augment: bool = False,
        noise_std: float = 0.01,
        scale_low: float = 0.9,
        scale_high: float = 1.1,
        time_shift_pct: float = 0.1,
    ):
        self.df = df.reset_index(drop=True)
        self.class_map = class_map
        self.signal_cols = signal_cols
        self.feature_cols = feature_cols
        self.label_col = label_col

        # Parametry augmentacji
        self.augment = augment
        self.noise_std = noise_std
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.time_shift_pct = time_shift_pct

    def __len__(self):
        return len(self.df)

    def _augment(self, sig: np.ndarray) -> np.ndarray:
        """Augmentacja 1D sygnału — wywoływana tylko w treningu."""
        if not self.augment:
            return sig

        # 1. Szum gaussowski
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, sig.shape)
            sig = sig + noise

        # 2. Skalowanie amplitudy
        if self.scale_low < 1.0 or self.scale_high > 1.0:
            scale = np.random.uniform(self.scale_low, self.scale_high)
            sig = sig * scale

        # 3. Przesunięcie w czasie (circular shift)
        if self.time_shift_pct > 0:
            max_shift = int(sig.shape[-1] * self.time_shift_pct)
            if max_shift > 0:
                shift = np.random.randint(-max_shift, max_shift + 1)
                sig = np.roll(sig, shift, axis=-1)

        return sig.astype(np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {}

        # Sygnały jako dict
        for col in self.signal_cols:
            sig = row[col]
            if isinstance(sig, np.ndarray) and sig.ndim == 1:
                sig = sig.reshape(len(sig) // 5000, 5000)

            if self.augment:
                sig = self._augment(sig)

            sample[col] = torch.tensor(sig, dtype=torch.float32)

        # Cechy demograficzne/tabularne
        if self.feature_cols:
            sample["features"] = torch.tensor(
                row[self.feature_cols].values.astype(float),
                dtype=torch.float32
            )

        # Label — używa self.label_col zamiast "scp_codes"
        label = self.class_map.get(row[self.label_col], 0)
        label = torch.tensor(label, dtype=torch.long)

        return sample, label