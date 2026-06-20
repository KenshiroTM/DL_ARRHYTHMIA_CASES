import pandas as pd
def diagnose_domain_shift(data: pd.DataFrame, n: int = 2000) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from scipy import signal as sp_signal
    import numpy as np

    first = data.head(n)['signal'].dropna()
    last = data.tail(n)['signal'].dropna()

    # Wyciągnij 1 odprowadzenie: (n, 1, 5000) -> (n, 5000)
    def flatten(x):
        s = np.stack(x.values)
        return s.squeeze(1) if s.ndim == 3 else s

    X1, X2 = flatten(first), flatten(last)
    print(f"Kształt po spłaszczeniu: {X1.shape}")

    # === 1. Statystyki ===
    for name, X in [("Baza 1", X1), ("Baza 2", X2)]:
        print(f"{name}: mean={X.mean():.4f}, std={X.std():.4f}") # średnia + odchylenie

    # === 2. PSD ===
    for name, X in [("Baza 1", X1), ("Baza 2", X2)]:
        f, psd = sp_signal.welch(X[:100], fs=500, nperseg=512, axis=1)
        low = psd[:, f < 5].mean() # zakres niskich częstotliwości
        mid = psd[:, (f >= 5) & (f < 40)].mean() # zakres średnich częstotliwości
        high = psd[:, f >= 40].mean() # zakres niskich częstotliwości
        print(f"{name} PSD: low={low:.2e}, mid={mid:.2e}, high={high:.2e}")

    # === 3. Klasyfikator ===
    def feats(X):
        return np.column_stack([
            X.mean(axis=1), X.std(axis=1), X.min(axis=1), X.max(axis=1),
            np.percentile(X, 25, axis=1), np.percentile(X, 75, axis=1),
            np.median(np.abs(np.diff(X, axis=1)), axis=1)
        ])

    X = np.vstack([feats(X1), feats(X2)])
    y = np.array([0] * len(X1) + [1] * len(X2))

    acc = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5).mean()

    print(f"\n{'=' * 40}")
    print(f"Rozróżnialność baz: {acc:.1%}")
    if acc > 0.75:
        print("WYNIK: Łatwo rozróżnialne → NIE ŁĄCZ")
    elif acc > 0.6:
        print("WYNIK: Słaba rozróżnialność → MOŻESZ spróbować")
    else:
        print("WYNIK: Nie do rozróżnienia → BEZPIECZNE")
    print(f"{'=' * 40}")