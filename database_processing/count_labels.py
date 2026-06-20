from pathlib import Path

import pandas as pd


def count_labels(df: pd.DataFrame, labels_count_csv: Path):
    rows = []
    for _, row in df.dropna(subset=["scp_codes"]).iterrows():
        for label in (l.strip() for l in str(row["scp_codes"]).split(",")):
            rows.append({"label": label})

    exploded = pd.DataFrame(rows)

    result = exploded.groupby("label").size().rename("count").reset_index()
    result = result.sort_values("count", ascending=False)

    result.to_csv(labels_count_csv, index=False)