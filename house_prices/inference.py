"""Model inference pipeline."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from house_prices import FEATURE_COLUMNS, MODELS_DIR
from house_prices.preprocess import preprocess


def make_predictions(filepath: str) -> np.ndarray:
    """Load the trained model and predict house prices for input data.

    Args:
        filepath: Path to the input CSV file containing the feature
            columns.

    Returns:
        Numpy array of predicted house prices, one per input row.
    """
    df = _load_input_data(filepath)
    X_processed = preprocess(df[FEATURE_COLUMNS], is_training=False)
    model = _load_trained_model()
    return model.predict(X_processed)


def _load_input_data(filepath: str) -> pd.DataFrame:
    """Read the input CSV file with a clear error if it is missing."""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    return pd.read_csv(filepath)


def _load_trained_model() -> LinearRegression:
    """Load the persisted model with a clear error if it is missing."""
    model_path = MODELS_DIR / "model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at {model_path}")
    return joblib.load(model_path)
