"""Shared preprocessing pipeline for training and inference."""
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from house_prices import (
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    MODELS_DIR,
)


def preprocess(df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
    """Preprocess feature columns for training or inference.

    Args:
        df: Raw dataframe containing the feature columns.
        is_training: When True, fit transformers on ``df`` and persist
            them in ``MODELS_DIR``. When False, load persisted
            transformers and only apply ``transform``.

    Returns:
        Dataframe of scaled continuous and one-hot encoded categorical
        features, ready to be fed to the model.
    """
    features = df[CONTINUOUS_FEATURES + CATEGORICAL_FEATURES].copy()
    features = _impute_missing_values(features, is_training)
    continuous = _scale_continuous(features, is_training)
    categorical = _encode_categorical(features, is_training)
    return pd.concat([continuous, categorical], axis=1)


def _impute_missing_values(
    df: pd.DataFrame, is_training: bool
) -> pd.DataFrame:
    """Fill missing values using persisted medians and modes."""
    medians, modes = _get_imputation_values(df, is_training)
    return df.fillna({**medians, **modes})


def _get_imputation_values(
    df: pd.DataFrame, is_training: bool
) -> Tuple[dict, dict]:
    """Compute and persist, or load, the imputation values."""
    if is_training:
        return _fit_and_save_imputation(df)
    medians = _safe_load(MODELS_DIR / "continuous_medians.joblib")
    modes = _safe_load(MODELS_DIR / "categorical_modes.joblib")
    return medians, modes


def _fit_and_save_imputation(df: pd.DataFrame) -> Tuple[dict, dict]:
    """Compute medians/modes from ``df`` and persist them to disk."""
    medians = df[CONTINUOUS_FEATURES].median().to_dict()
    modes = {col: df[col].mode()[0] for col in CATEGORICAL_FEATURES}
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(medians, MODELS_DIR / "continuous_medians.joblib")
    joblib.dump(modes, MODELS_DIR / "categorical_modes.joblib")
    return medians, modes


def _scale_continuous(
    df: pd.DataFrame, is_training: bool
) -> pd.DataFrame:
    """Standardize continuous features using a persisted scaler."""
    scaler = _get_scaler(df, is_training)
    transformed = scaler.transform(df[CONTINUOUS_FEATURES])
    return pd.DataFrame(
        transformed, columns=CONTINUOUS_FEATURES, index=df.index
    )


def _get_scaler(df: pd.DataFrame, is_training: bool) -> StandardScaler:
    """Either fit-and-save the scaler, or load it from disk."""
    if is_training:
        scaler = StandardScaler()
        scaler.fit(df[CONTINUOUS_FEATURES])
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
        return scaler
    return _safe_load(MODELS_DIR / "scaler.joblib")


def _encode_categorical(
    df: pd.DataFrame, is_training: bool
) -> pd.DataFrame:
    """One-hot encode categorical features using a persisted encoder."""
    encoder = _get_encoder(df, is_training)
    transformed = encoder.transform(df[CATEGORICAL_FEATURES])
    columns = encoder.get_feature_names_out(CATEGORICAL_FEATURES)
    return pd.DataFrame(transformed, columns=columns, index=df.index)


def _get_encoder(df: pd.DataFrame, is_training: bool) -> OneHotEncoder:
    """Either fit-and-save the encoder, or load it from disk."""
    if is_training:
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False
        )
        encoder.fit(df[CATEGORICAL_FEATURES])
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(encoder, MODELS_DIR / "encoder.joblib")
        return encoder
    return _safe_load(MODELS_DIR / "encoder.joblib")


def _safe_load(path: Path):
    """Load a joblib artifact with a clear error if missing."""
    if not path.exists():
        raise FileNotFoundError(f"Required artifact missing: {path}")
    return joblib.load(path)
