from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.config import Paths, ensure_directories
from src.data import make_submission, validate_raw_data, write_dataframe


def parse_args() -> argparse.Namespace:
    paths = Paths()
    parser = argparse.ArgumentParser(description="Generate a competition submission.")
    parser.add_argument("--model-path", type=Path, default=paths.final_model)
    parser.add_argument("--train-path", type=Path, default=paths.train)
    parser.add_argument("--test-path", type=Path, default=paths.test)
    parser.add_argument("--sample-submission-path", type=Path, default=paths.sample_submission)
    parser.add_argument("--output-path", type=Path, default=paths.latest_submission)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_directories()

    model = joblib.load(args.model_path)
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    sample_submission = pd.read_csv(args.sample_submission_path)
    validate_raw_data(train, test, sample_submission)

    predictions = model.predict(test)
    submission = make_submission(test, predictions)
    write_dataframe(submission, args.output_path)
    print(f"Wrote submission to {args.output_path}")


if __name__ == "__main__":
    main()

