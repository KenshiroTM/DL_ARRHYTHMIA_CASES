import ast
import os
import pandas as pd
import wfdb
from tqdm import tqdm
from config import *
from .deduplicate_records import deduplicate_records
from .count_labels import count_labels
from .preload_data import preload_data
from .filter_data_by_label import filter_data_by_label

tqdm.pandas(desc="Ładowanie...")

def process_ptbxl():
    if BINARY_DATA.exists():
        print("[ptbxl] Wczytuję istniejący parquet...")
        return pd.read_parquet(BINARY_DATA)

    csv_file = ptbxl / ptbxl_csv
    if not csv_file.exists():
        raise FileNotFoundError(f"Nie znaleziono csv pod: {csv_file}")

    df = pd.read_csv(csv_file)
    df = df[headers_ptbxl]

    if "filename_hr" in headers_ptbxl:
        df["filename_hr"] = df["filename_hr"].progress_apply(lambda x: str(ptbxl / x) + ".dat")
    if "scp_codes" in headers_ptbxl:
        df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
        df["scp_codes"] = df["scp_codes"].apply(lambda scp: _classify_threshold(scp, threshold))
    if "sex" in headers_ptbxl:
        df["sex"] = df["sex"].map(SEX_MAP)
    if "age" in headers_ptbxl:
        df = df[df["age"] <= 120]

    df = df.dropna()
    df = deduplicate_records(df, "patient_id")
    df["scp_codes"] = df["scp_codes"].apply(lambda x: x[0] if isinstance(x, list) else x)

    count_labels(df, processed_folder / "labels_ptbxl.csv")
    df.to_csv(BINARY_CSV, index=False)

    data = preload_data(df, leads, "filename_hr")
    data = filter_data_by_label(data, "scp_codes", bin_map)
    data.to_parquet(BINARY_DATA)

    return data

def process_arrhythmia():
    if ARRHYTHMIA_DATA.exists():
        print("[arrhythmia] Wczytuję istniejący parquet...")
        return pd.read_parquet(ARRHYTHMIA_DATA)

    csv_file = arrhythmia / arrhythmia_csv
    if not csv_file.exists():
        raise FileNotFoundError(f"Nie znaleziono csv pod: {csv_file}")

    condition_map = _load_condition_names(csv_file)
    header_paths = list(arrhythmia.rglob("*.hea"))
    print(f"Znaleziono {len(header_paths)} plików .hea")

    rows = []
    for path in header_paths:
        try:
            header = wfdb.rdheader(str(path.with_suffix("")))
            info = _parse_header(header, headers_arrhythmia, condition_map)
            info["path"] = str(path.with_suffix(".mat"))
            rows.append(info)
        except Exception as e:
            print(f"[SKIP] {path.name}: {e}")

    df = pd.DataFrame(rows)
    df = df.dropna()
    df = df[df["dx"].progress_apply(lambda x: len(x) == 1)]
    df["dx"] = df["dx"].map(lambda x: x[0])
    df = df.rename(columns={"dx": "scp_codes", "path": "filename_hr"})
    df["scp_codes"] = df["scp_codes"].map(lambda code: LABEL_MAP_TO_PTBXL.get(code.upper(), code.upper()))

    print(f"Wczytano: {len(df)} headerów")
    count_labels(df, processed_folder / "labels_arrhythmia.csv")
    df.to_csv(ARRHYTHMIA_CSV, index=False)

    data = preload_data(df, leads, "filename_hr")
    data = filter_data_by_label(data, "scp_codes", arr_map)
    data.to_parquet(ARRHYTHMIA_DATA)

    return data

def process_vitaldb():
    if ANOMALY_DATA.exists():
        print("[vitaldb] Wczytuję istniejący parquet...")
        return pd.read_parquet(ANOMALY_DATA)

    print("Wczytywanie clinical_data.csv...")
    clinical_df = pd.read_csv(VITALDB_CSV)

    # Znajdź caseidy które mają oba pliki (PPG i ECG)
    print("Szukanie par PPG+EKG...")
    ppg_files = {_get_caseid_from_file(f) for f in os.listdir(VITALDB_SIGNALS_DIR) if f.startswith("ppg_")}
    ecg_files = {_get_caseid_from_file(f) for f in os.listdir(VITALDB_SIGNALS_DIR) if f.startswith("ecg_")}
    valid_caseids = ppg_files & ecg_files  # tylko te co mają oba sygnały
    valid_caseids.discard(None)
    print(f"Znaleziono {len(valid_caseids)} par PPG+EKG")

    # usuwanie tych samych pacjentów
    clinical_df = deduplicate_records(clinical_df, "subjectid")

    # Filtruj clinical_data tylko do tych caseids
    df = clinical_df[clinical_df["caseid"].isin(valid_caseids)].copy()

    # Wyciągnij tylko potrzebne kolumny
    df = df[["caseid", "age", "sex", "preop_ecg"]].copy()

    # Mapuj płeć
    df["sex"] = df["sex"].map(SEX_MAP)

    # Mapuj etykiety EKG na nasze klasy
    df["scp_codes"] = df["preop_ecg"].map(VITALDB_LABEL_MAP)

    # Usuń wiersze bez etykiety (rzadkie arytmie których nie mapujemy)
    before = len(df)
    df = df.dropna(subset=["scp_codes", "age", "sex"])
    after = len(df)
    print(f"Usunięto {before - after} wierszy bez etykiety lub z brakującymi danymi")

    # Filtruj nierealistyczny wiek
    df = df[df["age"] <= 120]

    # Dodaj ścieżki do plików sygnałów
    df["ppg_path"] = df["caseid"].apply(lambda c: str(VITALDB_SIGNALS_DIR / f"ppg_case_{c}.dat"))
    df["ecg_path"] = df["caseid"].apply(lambda c: str(VITALDB_SIGNALS_DIR / f"ecg_case_{c}.dat"))

    # Usuń niepotrzebne kolumny i dopasuj format do merged.csv
    df = df[["age", "sex", "scp_codes", "ppg_path", "ecg_path"]]

    # Eksport do CSV
    df.to_csv(ANOMALY_CSV, index=False)
    print(f"\nEksport CSV: {ANOMALY_CSV}")

    # podmiana ścieżek na surowe dane (ppg i ecg)
    df = preload_data(df, leads, "ppg_path", 'ppg_signal')
    df = preload_data(df, leads, "ecg_path", 'ecg_signal')

    # Pokaż statystyki
    print(f"\nLiczba rekordów: {len(df)}")
    print(f"Rozkład klas:\n{df['scp_codes'].value_counts()}")
    print(f"Rozkład płci:\n{df['sex'].value_counts()}")
    print(f"Wiek - min: {df['age'].min()}, max: {df['age'].max()}, średnia: {df['age'].mean():.1f}")

    # Eksport do parquet
    df.to_parquet(ANOMALY_DATA)
    print(f"Eksport Parquet: {ANOMALY_DATA}")

    return df

# === POMOCNICZE ===

def _classify_threshold(scp_code: dict, threshold: int):
    above = [label for label, value in scp_code.items() if value >= threshold]
    if len(above) != 1:
        return float("nan")
    return "NORM" if above[0] == "NORM" else "ANORM"


def _get_caseid_from_file(filename: str) -> int | None:
    try:
        name = filename.replace("ppg_case_", "").replace("ecg_case_", "").replace(".dat", "")
        return int(name)
    except ValueError:
        return None


def _parse_header(header, fields: list[str], condition_map: dict) -> dict:
    info = {f: None for f in fields}
    if "dx" in info:
        info["dx"] = []

    for comment in header.comments:
        comment = comment.strip().lstrip("#")
        if "age" in fields and comment.startswith("Age:"):
            try:
                info["age"] = int(comment.replace(" ", "").split(":")[1])
            except ValueError:
                info["age"] = None
        elif "sex" in fields and comment.startswith("Sex:"):
            sex = comment.replace(" ", "").split(":")[1]
            info["sex"] = sex if sex in ("Male", "Female") else None
        elif "dx" in fields and comment.startswith("Dx:"):
            raw = comment.replace(" ", "").split(":")[1]
            converted = [condition_map.get(d, d) for d in raw.split(",") if d]
            info["dx"] = None if any(c.isdigit() for c in converted) else converted

    return info


def _load_condition_names(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path)
    return dict(zip(df["Snomed_CT"].astype(str), df["Acronym Name"]))