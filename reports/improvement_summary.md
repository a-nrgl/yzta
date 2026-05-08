# Improvement Summary

## Baseline

- Previous public leaderboard score reported by user: `1.20901`
- Previous saved local CV: `1.22007` RMSE, `0.97335` MAE, `0.70113` R2
- Previous model: weighted `VotingRegressor` with two one-hot HGB variants and `RidgeCV`
- Previous validation: `KFold(n_splits=5, shuffle=True, random_state=42)`

## Data And Metric Assumptions

- The docs do not specify an official metric, so local model selection used RMSE.
- MAE and R2 were tracked as secondary diagnostics.
- Train/test schemas match, IDs are unique, and no duplicated feature rows were found.
- Predictions are clipped to the observed target range `[0, 10]`.

## Experiments Tried

- Rebuilt the previous saved ensemble as the baseline.
- Tested richer sleep, stress, activity, health, and category-combination features.
- Compared one-hot HGB variants with scaled and unscaled numeric preprocessing.
- Tested scikit-learn native-categorical `HistGradientBoostingRegressor` variants.
- Optimized non-negative OOF ensemble weights using only training-fold predictions.

Full results are in `reports/experiment_results.csv`.

## Best Validated Model

Final model: `improved_ensemble`

Components:

- `RidgeCV` with improved engineered features and one-hot categorical encoding
- three one-hot HGB variants using improved features
- three native-categorical HGB variants using current/improved features

Rounded final weights:

- ridge plus features: `0.41`
- HGB main one-hot: `0.02`
- HGB regularized one-hot: `0.10`
- HGB slow one-hot: `0.04`
- HGB native current features: `0.19`
- HGB native small-leaf improved features: `0.17`
- HGB native slow improved features: `0.07`

## Best Local CV

- Best OOF weight-search estimate: `1.21704` RMSE
- Final CLI-verified model: `1.21712` RMSE, `0.97109` MAE, `0.70257` R2
- Fold RMSE std: `0.01079`

This improves local CV by about `0.00295` RMSE versus the previous saved ensemble.

## Final Submission

- Final path: `data/submissions/submission_improved.csv`
- Rows: `24,000`
- Columns: `id`, `bilissel_performans_skoru`
- IDs match `data/raw/test_x.csv`
- No missing predictions
- Prediction range: `[0, 10]`

## Next Ideas

- Try CatBoost or LightGBM if a Python-compatible wheel is available in the runtime.
- Re-check public/private correlation after the new submission; local CV is slightly pessimistic versus the reported previous public score.
- Try a small, nested validation for ensemble weights if more runtime is available.
- Explore target-bound modeling for clipped values near `0` and `10`.
