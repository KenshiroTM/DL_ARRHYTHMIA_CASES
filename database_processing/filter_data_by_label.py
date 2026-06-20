import pandas as pd
def filter_data_by_label(df: pd.DataFrame, label_column: str, n_classes: dict) -> pd.DataFrame:

    # Czy kolumna dana istnieje
    if label_column not in df.columns:
        raise ValueError(f"DataFrame nie zawiera kolumny '{label_column}'.")

    # Czy dict jest pusty
    if not n_classes:
        raise ValueError("Słownik 'n_classes' nie może być pusty.")

    allowed_labels = set(n_classes.keys())
    filtered_df = df[df[label_column].isin(allowed_labels)].copy() # konwert na stringi kluczy
    return filtered_df