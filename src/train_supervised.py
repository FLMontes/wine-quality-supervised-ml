from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))
from data_loader import load_wine_quality


def run_classification(df: pd.DataFrame) -> None:
    """Train a baseline classifier to predict wine type."""
    X = df.drop(columns=["wine_type", "quality"])
    y = df["wine_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Classification — Wine Type")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Weighted F1-score: {f1_score(y_test, predictions, average='weighted'):.4f}")


def run_regression(df: pd.DataFrame) -> None:
    """Train a baseline regressor to estimate wine quality."""
    X = df.drop(columns=["quality", "wine_type"])
    y = df["quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("
Regression — Wine Quality")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
    print(f"R²: {r2_score(y_test, predictions):.4f}")


if __name__ == "__main__":
    data = load_wine_quality()
    run_classification(data)
    run_regression(data)
