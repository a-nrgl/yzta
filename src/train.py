from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import joblib

from src.config import N_SPLITS, RANDOM_STATE, Paths, ensure_directories
from src.data import load_raw_data, make_submission, split_features_target, write_dataframe
from src.evaluate import cross_validate_pipeline, write_metrics
from src.features import add_features
from src.models import build_model_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the regression model and create submission.")
    parser.add_argument(
        "--model-name",
        default="ensemble",
        choices=["ridge", "hgb", "hgb_regularized", "extra_trees", "ensemble"],
        help="Estimator recipe to train.",
    )
    parser.add_argument("--n-splits", type=int, default=N_SPLITS, help="Number of CV folds.")
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-cv", action="store_true", help="Fit final model without CV.")
    parser.add_argument(
        "--save-processed",
        action="store_true",
        help="Write engineered train/test feature tables to data/processed.",
    )
    parser.add_argument(
        "--submission-name",
        default=None,
        help="Optional explicit submission filename under data/submissions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths()
    ensure_directories(paths)

    started = perf_counter()
    train, test, _ = load_raw_data(paths)
    X, y = split_features_target(train)

    if args.save_processed:
        write_dataframe(add_features(X), paths.data_processed / "train_features.csv")
        write_dataframe(add_features(test), paths.data_processed / "test_features.csv")

    pipeline = build_model_pipeline(
        sample_features=X,
        model_name=args.model_name,
        random_state=args.random_state,
    )

    metrics: dict[str, object] = {
        "model_name": args.model_name,
        "random_state": args.random_state,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "target": "bilissel_performans_skoru",
        "prediction_bounds": [0.0, 10.0],
    }

    if not args.skip_cv:
        cv_started = perf_counter()
        oof, fold_metrics, cv_metrics = cross_validate_pipeline(
            pipeline,
            X,
            y,
            n_splits=args.n_splits,
            random_state=args.random_state,
        )
        write_dataframe(oof, paths.oof_predictions)
        write_dataframe(fold_metrics, paths.fold_metrics)
        metrics["cv"] = cv_metrics
        metrics["cv_runtime_seconds"] = round(perf_counter() - cv_started, 3)

    final_started = perf_counter()
    pipeline.fit(X, y)
    paths.final_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, paths.final_model)
    metrics["final_fit_runtime_seconds"] = round(perf_counter() - final_started, 3)

    predictions = pipeline.predict(test)
    submission = make_submission(test, predictions)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_submission_path = paths.submissions / f"submission_{args.model_name}_{timestamp}.csv"
    write_dataframe(submission, timestamped_submission_path)
    write_dataframe(submission, paths.latest_submission)
    if args.submission_name:
        write_dataframe(submission, paths.submissions / args.submission_name)

    metrics["model_path"] = str(paths.final_model)
    metrics["submission_path"] = str(timestamped_submission_path)
    metrics["latest_submission_path"] = str(paths.latest_submission)
    metrics["total_runtime_seconds"] = round(perf_counter() - started, 3)
    write_metrics(metrics, paths.metrics)

    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
