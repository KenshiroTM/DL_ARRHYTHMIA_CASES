import ast
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _parse_codes(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
    return df


def _get_top_labels(df: pd.DataFrame, codes_col: str, top_n: int) -> list:
    return [l for l, _ in Counter(df[codes_col]).most_common(top_n)]


def _save_and_close(fig: plt.Figure, output_path: Path) -> plt.Figure:
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close()
    return fig


def plot_hist(df: pd.DataFrame, col: str, output_path: Path, title: str = "", bins: int = 30, figsize: tuple = (10, 5)) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(df[col].dropna(), bins=bins, color="steelblue", edgecolor="black")
    ax.set_title(title or f"Rozkład {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Liczba rekordów")
    return _save_and_close(fig, output_path)


def plot_pie(df: pd.DataFrame, col: str, output_path: Path, title: str = "", figsize: tuple = (6, 6)) -> plt.Figure:
    counts = df[col].value_counts()
    fig, ax = plt.subplots(figsize=figsize)
    ax.pie(counts, labels=counts.index, autopct="%1.1f%%", colors=["steelblue", "salmon"])
    ax.set_title(title or f"Rozkład {col}")
    return _save_and_close(fig, output_path)


def plot_label_bar(df: pd.DataFrame, codes_col: str, output_path: Path, title: str = "", top_n: int = 20, figsize: tuple = (12, 6)) -> plt.Figure:
    df = _parse_codes(df, codes_col)
    all_labels = Counter(df[codes_col])
    top = all_labels.most_common(top_n)
    labels, counts = zip(*top)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, counts, color="steelblue", edgecolor="black")
    ax.set_title(title or f"Top {top_n} labeli")
    ax.set_xlabel("Label")
    ax.set_ylabel("Liczba rekordów")
    plt.xticks(rotation=45, ha="right")
    return _save_and_close(fig, output_path)


def plot_boxplot_per_class(df: pd.DataFrame, value_col: str, codes_col: str, output_path: Path, title: str = "", top_n: int = 10, figsize: tuple = (14, 6)) -> plt.Figure:
    df = _parse_codes(df, codes_col)
    top_labels = _get_top_labels(df, codes_col, top_n)

    rows = [{"label": row[codes_col], "value": row[value_col]} for _, row in df.iterrows() if row[codes_col] in top_labels]
    plot_df = pd.DataFrame(rows)
    groups = [plot_df[plot_df["label"] == l]["value"].dropna() for l in top_labels]

    fig, ax = plt.subplots(figsize=figsize)
    ax.boxplot(groups, labels=top_labels)
    ax.set_title(title or f"Rozkład {value_col} per klasa")
    ax.set_xlabel("Klasa")
    ax.set_ylabel(value_col)
    plt.xticks(rotation=45, ha="right")
    return _save_and_close(fig, output_path)


def plot_stacked_bar_per_class(df: pd.DataFrame, group_col: str, codes_col: str, output_path: Path, title: str = "", top_n: int = 10, figsize: tuple = (14, 6)) -> plt.Figure:
    df = _parse_codes(df, codes_col)
    top_labels = _get_top_labels(df, codes_col, top_n)

    rows = [{"label": row[codes_col], "group": row[group_col]} for _, row in df.iterrows() if row[codes_col] in top_labels]
    plot_df = pd.DataFrame(rows)
    pivot = plot_df.groupby(["label", "group"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", ax=ax, color=["salmon", "steelblue"], edgecolor="black")
    ax.set_title(title or f"Rozkład {group_col} per klasa")
    ax.set_xlabel("Klasa")
    ax.set_ylabel("Liczba rekordów")
    plt.xticks(rotation=45, ha="right")
    return _save_and_close(fig, output_path)