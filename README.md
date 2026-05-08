# YZTA 2026 Datathon Regression Solution

This project predicts `bilissel_performans_skoru` for the competition test set. The pipeline is built around reproducible feature engineering, cross-validation, model persistence, and submission generation.

## Project Layout

```text
data/raw/              Original train, test, and sample submission files
data/processed/        Optional engineered feature exports
data/submissions/      Generated submission files
docs/                  Competition and dataset notes
notebooks/             EDA, modeling, tuning, and submission notebooks
src/                   Reusable training and prediction code
models/                Saved fitted model artifacts
reports/               Metrics, fold scores, and OOF predictions
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the default ensemble, run five-fold CV, save the model, and write submissions:

```bash
python -m src.train
```

Train the current improved ensemble and write the competition file:

```bash
python -m src.train --model-name improved_ensemble --submission-name submission_improved.csv
```

Useful alternatives:

```bash
python -m src.train --model-name improved_ensemble
python -m src.train --model-name hgb
python -m src.train --model-name ridge
python -m src.train --skip-cv
python -m src.train --save-processed
python -m src.predict --model-path models/final_model.joblib
```

Main outputs:

- `models/final_model.joblib`
- `reports/metrics.json`
- `reports/fold_metrics.csv`
- `reports/oof_predictions.csv`
- `data/submissions/submission_latest.csv`

Latest validated run:

- model: `improved_ensemble`
- five-fold OOF RMSE: `1.2171`
- five-fold OOF MAE: `0.9711`
- five-fold OOF R2: `0.7026`
- final submission: `data/submissions/submission_improved.csv`

## Modeling Approach

The original default model is a weighted ensemble of two `HistGradientBoostingRegressor` variants and a regularized ridge model. The improved model adds richer sleep/lifestyle features plus native-categorical HGB variants. The preprocessing pipeline handles:

- median imputation plus missing indicators for numeric columns
- constant imputation and one-hot encoding for categorical columns
- engineered sleep quality, fragmentation, stress, activity, health, and interaction features
- prediction clipping to the observed target range `[0, 10]`

The final submission is generated with the exact two-column format:

```text
id,bilissel_performans_skoru
```

Run `python -m src.evaluate` after training to print the latest metrics JSON.
