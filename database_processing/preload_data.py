import os

import numpy as np
import pandas as pd
import scipy
from pandas.core.interchange.dataframe_protocol import DataFrame
from tqdm import tqdm

tqdm.pandas(desc="Wczytywanie plików dat/mat...")

def preload_data(df: DataFrame, leads: list, path_col: str, new_col: str = 'signal') -> pd.DataFrame:

    print(f"Rozpoczynam wczytywanie {len(leads)} odprowadzeń dla {len(df)} wierszy...")
    df[new_col] = df[path_col].progress_apply(lambda x: extract_signal(x,leads)) # wyciągnięcie sygnałów

    return df

def extract_signal(path, leads):
    if not isinstance(path, str) or not os.path.exists(path):
        return None

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.mat':
            data = scipy.io.loadmat(path)
            # Szukamy głównego klucza z danymi (omijamy metadane matlaba)
            key = [k for k in data.keys() if not k.startswith('__')][0]
            signal = data[key]  # Macierz (leads, samples)
            return (signal[leads, :] / 1000).flatten().astype(np.float32)

        elif ext == '.dat':
            # --- VitalDB (float32, 1 lead) ---
            if 'vitaldb' in path.lower():
                raw_data = np.fromfile(path, dtype=np.float32)
                signal = raw_data.reshape(1, -1)  # (1, samples)
                return (signal[leads, :]).flatten().astype(np.float32)

            # --- Standardowe .dat (int16, 12 leads) ---
            else:
                raw_data = np.fromfile(path, dtype=np.int16)
                num_leads = 12
                signal = raw_data.reshape(-1, num_leads).T
                return (signal[leads, :] / 1000).flatten().astype(np.float32)

    except Exception as e:
        print(f"Błąd w pliku {path}: {e}")
        return None