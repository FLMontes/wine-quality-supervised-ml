from __future__ import annotations

import pandas as pd

RED_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
WHITE_WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"


def load_wine_quality() -> pd.DataFrame:
    """Load and combine red and white wine quality datasets from UCI."""
    red = pd.read_csv(RED_WINE_URL, sep=";")
    white = pd.read_csv(WHITE_WINE_URL, sep=";")

    red["wine_type"] = "red"
    white["wine_type"] = "white"

    return pd.concat([red, white], ignore_index=True)
