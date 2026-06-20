import pandas as pd

def deduplicate_records(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=[col], keep="first")
    df = df.drop(columns=col)
    print(f"Po deduplikacji: {len(df)} rekordów")
    return df