from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold

from src.config import ID_COLUMN, N_SPLITS, RANDOM_STATE, TARGET_COLUMN
from src.data import clip_predictions


def score_predictions(y_true: pd.Series | np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    clipped = clip_predictions(predictions)
    return {
        "rmse": float(root_mean_squared_error(y_true, clipped)),
        "mae": float(mean_absolute_error(y_true, clipped)),
        "r2": float(r2_score(y_true, clipped)),
    }


def cross_validate_pipeline(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run K-fold CV and return OOF predictions, fold metrics, and summary metrics."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_predictions = np.zeros(len(X), dtype=float)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
        estimator = clone(pipeline)
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        estimator.fit(X_train, y_train)
        valid_predictions = clip_predictions(estimator.predict(X_valid))
        oof_predictions[valid_idx] = valid_predictions
        fold_score = score_predictions(y_valid, valid_predictions)
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(train_idx)),
                "valid_rows": int(len(valid_idx)),
                **fold_score,
            }
        )

    fold_metrics = pd.DataFrame(fold_rows)
    oof = pd.DataFrame(
        {
            ID_COLUMN: X[ID_COLUMN].to_numpy(),
            TARGET_COLUMN: y.to_numpy(),
            "prediction": oof_predictions,
            "residual": y.to_numpy() - oof_predictions,
        }
    )
    summary = score_predictions(y, oof_predictions)
    summary.update(
        {
            "n_splits": int(n_splits),
            "rows": int(len(X)),
            "fold_rmse_std": float(fold_metrics["rmse"].std(ddof=0)),
            "fold_mae_std": float(fold_metrics["mae"].std(ddof=0)),
        }
    )
    return oof, fold_metrics, summary


def write_metrics(metrics: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return path


def main() -> None:
    path = Path("reports/metrics.json")
    if not path.exists():
        raise SystemExit("No metrics file found. Run `python -m src.train` first.")
    print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

