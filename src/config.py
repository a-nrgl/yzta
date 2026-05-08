from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"
    submissions: Path = PROJECT_ROOT / "data" / "submissions"
    models: Path = PROJECT_ROOT / "models"
    reports: Path = PROJECT_ROOT / "reports"
    train: Path = data_raw / "train.csv"
    test: Path = data_raw / "test_x.csv"
    sample_submission: Path = data_raw / "sample_submission.csv"
    final_model: Path = models / "final_model.joblib"
    metrics: Path = reports / "metrics.json"
    fold_metrics: Path = reports / "fold_metrics.csv"
    oof_predictions: Path = reports / "oof_predictions.csv"
    latest_submission: Path = submissions / "submission_latest.csv"


TARGET_COLUMN = "bilissel_performans_skoru"
ID_COLUMN = "id"
TARGET_MIN = 0.0
TARGET_MAX = 10.0
RANDOM_STATE = 42
N_SPLITS = 5


def ensure_directories(paths: Paths = Paths()) -> None:
    """Create output directories used by the training and prediction scripts."""
    for path in (
        paths.data_processed,
        paths.submissions,
        paths.models,
        paths.reports,
    ):
        path.mkdir(parents=True, exist_ok=True)

