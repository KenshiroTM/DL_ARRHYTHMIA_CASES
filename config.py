from enum import Enum
from pathlib import Path

# Foldery
db_folder = "databases"
model_folder = "models"
charts_dir = Path("charts")
weights_folder = Path(model_folder) / "weights"
results_folder = Path("results")
processed_folder = Path("databases_processed")

# Baza arrhythmia
arrhythmia = Path(db_folder) / "a-large-scale-12-lead-electrocardiogram-database-for-arrhythmia-study-1.0.0"
arrhythmia_csv = "ConditionNames_SNOMED-CT.csv"
headers_arrhythmia = ["age", "sex", "dx"]

# PTB-XL
ptbxl = Path(db_folder) / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
ptbxl_csv = "ptbxl_database.csv"
headers_ptbxl = ["age", "sex", "patient_id", "scp_codes", "filename_hr"]
threshold = 80

# Baza vitaldb
VITALDB_SIGNALS_DIR   = Path("databases/vitaldb/signals")
VITALDB_CSV  = Path("databases/vitaldb/csv/clinical_data.csv")

# Eksporty danych
BINARY_CSV= processed_folder / Path("binary.csv")
BINARY_DATA = processed_folder / Path("binary_dataset.parquet")
ARRHYTHMIA_CSV = processed_folder / Path("arrhythmia.csv")
ARRHYTHMIA_DATA = processed_folder / Path("arrhythmia_dataset.parquet")
ANOMALY_CSV = processed_folder / Path("vitaldb.csv")
ANOMALY_DATA = processed_folder / Path("vitaldb_dataset.parquet")

# Mapy labeli
arr_labels = ['SBRAD', 'SR', 'STACH', 'AFLT', 'SARRH']
binary_label = ['NORM', 'ANORM']

arr_map = {label: idx for idx, label in enumerate(arr_labels)}
bin_map = {label: idx for idx, label in enumerate(binary_label)}

SEX_MAP = {"M": "Male", "F": "Female", 1: "Male", 0: "Female"}

LABEL_MAP_TO_PTBXL = {
    "SR": "SR", "AFIB": "AFIB", "LVH": "LVH", "RVH": "RVH",
    "WPW": "WPW", "1AVB": "1AVB", "2AVB": "2AVB", "3AVB": "3AVB",
    "SB": "SBRAD", "ST": "STACH", "AF": "AFLT", "SA": "SARRH",
    "SVT": "SVTAC", "APB": "PAC", "VPB": "PVC", "PRIE": "LPR",
    "LBBB": "CLBBB", "RBBB": "CRBBB", "AVNRT": "PSVT", "AVRT": "PSVT",
    "AT": "SVTAC", "VB": "BIGU", "ABI": "BIGU",
}

# etykiety vitaldb
VITALDB_LABEL_MAP = {
    "Normal Sinus Rhythm": "NORM",
    "Atrial fibrillation": "AFIB",
    "Atrial fibrillation with rapid ventricular response": "AFIB",
    "Atrial fibrillation with slow ventricular response": "AFIB",
    "Atrial fibrillation with premature ventricular or aberrantly conducted complexes": "AFIB",
    "Atrial fibrillation, Right bundle branch block": "AFIB",
    "Atrial fibrillation with premature ventricular, Incomplete left bundle block": "AFIB",
    "Atrial flutter with 2:1 A-V conduction": "AFIB",
    "Atrial flutter with variable A-V block": "AFIB",
}

# Odprowadzenia
leads = [0]
n_leads = len(leads)

# Podział danych
train_size = 0.8
val_size = 0.1
seed = 42

# Ścieżki do wag
ECG_PRETRAINED_WEIGHTS = weights_folder / "1_lead_ECGFounder.pth"

#typy modelu
class DatasetType(Enum):
    BINARY = "binary"
    ARRHYTHMIA = "arrhythmia"