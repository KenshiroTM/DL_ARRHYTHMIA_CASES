import pandas as pd
from sklearn.model_selection import train_test_split


def split_data_encoder(
        df: pd.DataFrame,
        train_size: float = 0.8,
        val_size: float = 0.1,
        seed: int = 42,
        normal_label: str = "NORM",
        label_col: str = "scp_codes",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """
    Podział danych dla autoenkodera.

    Train: wyłącznie klasa 'normal_label' (np. "NORM").
    Val / Test: wszystkie klasy, anomalie podzielone ~równo między val i test.
    """

    def _get_primary_label(x):
        """Wyciąga pierwszy label z listy lub zwraca wartość bezpośrednio."""
        return x[0] if isinstance(x, list) else x

    def _print_class_stats(df_split, name):
        """Printuje liczbę próbek per klasa."""
        labels = df_split[label_col].apply(_get_primary_label)
        counts = labels.value_counts().sort_index()
        print(f"\n--- {name} ({len(df_split)} próbek) ---")
        for cls, cnt in counts.items():
            marker = " ← normal_label" if cls == normal_label else ""
            print(f"  {cls}: {cnt}{marker}")
        return counts

    # Główny label
    df = df.copy()
    df["_primary_label"] = df[label_col].apply(_get_primary_label)

    # --- 1. Train: tylko NORM ---
    norm_mask = df["_primary_label"] == normal_label
    norm_df = df[norm_mask].drop(columns=["_primary_label"])
    anom_df = df[~norm_mask].copy()

    if anom_df.empty:
        raise ValueError("Brak anomalii w zbiorze! Autoenkoder wymaga anomalii do ewaluacji.")

    # Train z NORM
    if train_size < 1.0:
        train_df, _ = train_test_split(norm_df, train_size=train_size, random_state=seed)
    else:
        train_df = norm_df

    # --- 2. Anomalie: podziel równo na val i test ---
    anom_df = anom_df.drop(columns=["_primary_label"])

    if len(anom_df) >= 2:
        anom_val, anom_test = train_test_split(
            anom_df,
            train_size=0.5,
            random_state=seed
        )
    else:
        # Tylko 1 anomalia — idzie do test
        anom_val = anom_df.iloc[0:0]  # pusty
        anom_test = anom_df

    # --- 3. NORM-y poza train: podziel na val i test ---
    remaining_norm = norm_df.drop(train_df.index)

    if remaining_norm.empty:
        raise ValueError("Brak NORM-ów poza train — zmniejsz train_size")

    # Proporcja NORM w val vs test (zgodnie z val_size)
    total_val_test = val_size + (1 - train_size - val_size)
    val_norm_ratio = val_size / total_val_test if total_val_test > 0 else 0.5

    norm_val, norm_test = train_test_split(
        remaining_norm,
        train_size=val_norm_ratio,
        random_state=seed,
    )

    # --- 4. Złóż val i test ---
    val_df = pd.concat([norm_val, anom_val], ignore_index=True)
    test_df = pd.concat([norm_test, anom_test], ignore_index=True)

    # Wyczyść indeksy
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # --- 5. Printy ---
    print("=" * 50)
    print("PODZIAŁ DANYCH — AUTOENKODER")
    print("=" * 50)

    _print_class_stats(train_df, "TRAIN")
    _print_class_stats(val_df, "VAL")
    if val_size > 0:
        _print_class_stats(test_df, "TEST")

    # Podsumowanie anomalii
    total_anom = len(anom_df)
    anom_in_val = len(anom_val)
    anom_in_test = len(anom_test)
    print(f"\n--- Podsumowanie anomalii ---")
    print(f"  Razem anomalii: {total_anom}")
    print(f"  W val:  {anom_in_val} ({anom_in_val / total_anom * 100:.1f}%)")
    print(f"  W test: {anom_in_test} ({anom_in_test / total_anom * 100:.1f}%)")
    print("=" * 50)

    if val_size == 0:
        return train_df, test_df, None

    return train_df, val_df, test_df