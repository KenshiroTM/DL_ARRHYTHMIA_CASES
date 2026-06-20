import os
import time

import pandas as pd
import vitaldb.dataset as vdb

# ── Ustawienia ────────────────────────────────────────────
SAMPLE_RATE    = 500   # Hz
DURATION_S     = 10    # sekund segmentu
START_OFFSET_S = 900   # pomiń pierwsze N sekund
MAX_CASES      = None  # None = wszystkie; np. 20 = tylko do testów
# ─────────────────────────────────────────────────────────

N_SAMPLES   = DURATION_S     * SAMPLE_RATE
OFFSET_SAMP = START_OFFSET_S * SAMPLE_RATE
API_URL     = "https://api.vitaldb.net"

os.makedirs("databases/vitaldb/signals", exist_ok=True)
os.makedirs("databases/vitaldb/csv",     exist_ok=True)

# ── CSV: pobierz wprost przez requests (omija bug w vdb) ─
print("Pobieranie plików CSV...")
for endpoint, filename in [("cases", "clinical_data"), ("labs", "lab_data"), ("trks", "tracks_info")]:
    df = pd.read_csv(f"{API_URL}/{endpoint}")
    df.to_csv(f"databases/vitaldb/csv/{filename}.csv", index=False)
    print(f"  {filename}.csv  →  {len(df):,} wierszy")

# ── Sygnały PPG + ECG → surowe .dat (float32) ────────────
print(f"\nPobieranie sygnałów  |  offset: {START_OFFSET_S}s  |  segment: {DURATION_S}s  |  {N_SAMPLES} próbek @ {SAMPLE_RATE} Hz\n")

case_ids = vdb.find_cases(["SNUADC/PLETH", "SNUADC/ECG_II"])
if MAX_CASES:
    case_ids = case_ids[:MAX_CASES]

for i, caseid in enumerate(case_ids, 1):
    ppg_path = f"databases/vitaldb/signals/ppg_case_{caseid}.dat"
    ecg_path = f"databases/vitaldb/signals/ecg_case_{caseid}.dat"

    if os.path.exists(ppg_path) and os.path.exists(ecg_path):
        print(f"  [{i}/{len(case_ids)}] case {caseid} – pominięto (już istnieje)")
        continue

    try:
        data = vdb.load_case(caseid, ["SNUADC/PLETH", "SNUADC/ECG_II"], interval=1/SAMPLE_RATE)

        if data is None or data.shape[0] < OFFSET_SAMP + N_SAMPLES:
            total_s = 0 if data is None else data.shape[0] / SAMPLE_RATE
            raise ValueError(f"za krótki sygnał ({total_s:.0f}s, potrzeba {START_OFFSET_S + DURATION_S}s)")

        seg = data[OFFSET_SAMP : OFFSET_SAMP + N_SAMPLES]
        seg[:, 0].astype("<f4").tofile(ppg_path)
        seg[:, 1].astype("<f4").tofile(ecg_path)
        print(f"  [{i}/{len(case_ids)}] case {caseid} – OK")

    except Exception as e:
        print(f"  [{i}/{len(case_ids)}] case {caseid} – pominięto ({e})")

    time.sleep(0.5)

print(f"\nGotowe! Dane w: databases/vitaldb/")
print(f"Odczyt: np.fromfile('ppg_case_1.dat', dtype='<f4')  →  {N_SAMPLES} próbek @ {SAMPLE_RATE} Hz")