"""Model training pipeline."""
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from house_prices import FEATURE_COLUMNS, LABEL_COLUMN, MODELS_DIR
from house_prices.preprocess import preprocess


def build_model(filepath: str) -> dict[str, float]:
    """Train a linear regression model and return evaluation metrics.

    Args:
        filepath: Path to the training CSV file.

    Returns:
        Dictionary mapping metric names (e.g. ``"rmsle"``) to their
        floating-point values on the held-out validation split.
    """
    X_train, X_valid, y_train, y_valid = _load_and_split(filepath)
    model = _fit_and_save_model(X_train, y_train)
    y_pred = model.predict(preprocess(X_valid, is_training=False))
    return {"rmsle": _compute_rmsle(y_valid, y_pred)}


def _load_and_split(
    filepath: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the training CSV and split into train/validation sets."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Training file not found: {filepath}")
    df = pd.read_csv(filepath)
    return train_test_split(
        df[FEATURE_COLUMNS], df[LABEL_COLUMN],
        test_size=0.2, random_state=42,
    )


def _fit_and_save_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> LinearRegression:
    """Preprocess training data, fit linear regression, persist it."""
    X_processed = preprocess(X_train, is_training=True)
    model = LinearRegression()
    model.fit(X_processed, y_train)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "model.joblib")
    return model


def _compute_rmsle(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """Root Mean Squared Logarithmic Error, predictions clipped to >= 1."""
    y_pred_clipped = np.maximum(y_pred, 1)
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred_clipped))
    return round(float(rmsle), 2)
