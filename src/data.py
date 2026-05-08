from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ID_COLUMN, TARGET_COLUMN, Paths


def load_raw_data(paths: Paths = Paths()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and sample submission files."""
    train = pd.read_csv(paths.train)
    test = pd.read_csv(paths.test)
    sample_submission = pd.read_csv(paths.sample_submission)
    validate_raw_data(train, test, sample_submission)
    return train, test, sample_submission


def validate_raw_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> None:
    """Validate schema assumptions that the pipeline depends on."""
    required_submission_columns = [ID_COLUMN, TARGET_COLUMN]
    missing_submission_cols = [
        col for col in required_submission_columns if col not in sample_submission.columns
    ]
    if missing_submission_cols:
        raise ValueError(f"Sample submission is missing columns: {missing_submission_cols}")

    if TARGET_COLUMN not in train.columns:
        raise ValueError(f"Train data is missing target column: {TARGET_COLUMN}")
    if TARGET_COLUMN in test.columns:
        raise ValueError(f"Test data should not contain target column: {TARGET_COLUMN}")
    if ID_COLUMN not in train.columns or ID_COLUMN not in test.columns:
        raise ValueError(f"Train and test data must both contain id column: {ID_COLUMN}")

    feature_columns = [col for col in train.columns if col != TARGET_COLUMN]
    missing_in_test = sorted(set(feature_columns) - set(test.columns))
    extra_in_test = sorted(set(test.columns) - set(feature_columns))
    if missing_in_test or extra_in_test:
        raise ValueError(
            "Train/test feature mismatch. "
            f"Missing in test: {missing_in_test}; extra in test: {extra_in_test}"
        )

    if not train[ID_COLUMN].is_unique:
        raise ValueError("Train ids are not unique.")
    if not test[ID_COLUMN].is_unique:
        raise ValueError("Test ids are not unique.")
    if train[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains missing values.")


def split_features_target(train: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split raw training data into features and target."""
    return train.drop(columns=[TARGET_COLUMN]), train[TARGET_COLUMN]


def clip_predictions(predictions: np.ndarray | pd.Series) -> np.ndarray:
    """Clip predictions to the observed target bounds used by the competition."""
    return np.clip(np.asarray(predictions, dtype=float), 0.0, 10.0)


def make_submission(test: pd.DataFrame, predictions: np.ndarray | pd.Series) -> pd.DataFrame:
    """Create a submission DataFrame in the required column order."""
    clipped = clip_predictions(predictions)
    if len(clipped) != len(test):
        raise ValueError(f"Prediction length {len(clipped)} does not match test length {len(test)}")
    return pd.DataFrame({ID_COLUMN: test[ID_COLUMN].to_numpy(), TARGET_COLUMN: clipped})


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    """Write a DataFrame as CSV and return the output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path

