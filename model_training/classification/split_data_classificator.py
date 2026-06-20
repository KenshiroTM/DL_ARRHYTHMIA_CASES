import pandas as pd
from sklearn.model_selection import train_test_split

def split_data_classificator(df: pd.DataFrame, train_size: float = 0.8, val_size: float = 0.1, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    stratify = df["scp_codes"].apply(lambda x: x[0] if isinstance(x, list) else x)

    train_df, temp_df = train_test_split(df, train_size=train_size, random_state=seed, stratify=stratify)

    if val_size == 0:
        print(f"Train: {len(train_df)}, Val: {len(temp_df)}")
        return train_df, temp_df, None

    val_ratio = val_size / (1 - train_size)
    stratify_temp = temp_df["scp_codes"].apply(lambda x: x[0] if isinstance(x, list) else x)
    val_df, test_df = train_test_split(temp_df, train_size=val_ratio, random_state=seed, stratify=stratify_temp)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df