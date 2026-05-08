from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.features import make_feature_transformer, make_preprocessor


def make_regressor(model_name: str = "ensemble", random_state: int = RANDOM_STATE):
    """Create a regression estimator by name."""
    model_name = model_name.lower()

    ridge = RidgeCV(alphas=np.logspace(-2, 4, 25))

    hgb_main = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=1100,
        learning_rate=0.03,
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=0.05,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=random_state,
    )
    hgb_regularized = HistGradientBoostingRegressor(
        loss="squared_error",
        max_iter=800,
        learning_rate=0.045,
        max_leaf_nodes=15,
        min_samples_leaf=35,
        l2_regularization=0.08,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=random_state + 17,
    )

    if model_name == "ridge":
        return ridge
    if model_name == "hgb":
        return hgb_main
    if model_name == "hgb_regularized":
        return hgb_regularized
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=350,
            max_features=0.75,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "ensemble":
        return VotingRegressor(
            estimators=[
                ("hgb_main", hgb_main),
                ("hgb_regularized", hgb_regularized),
                ("ridge", ridge),
            ],
            weights=[0.25, 0.30, 0.45],
            n_jobs=1,
        )

    raise ValueError(
        f"Unknown model_name={model_name!r}. "
        "Use one of: ridge, hgb, hgb_regularized, extra_trees, ensemble."
    )


def build_model_pipeline(
    sample_features: pd.DataFrame,
    model_name: str = "ensemble",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build the full feature engineering, preprocessing, and model pipeline."""
    return Pipeline(
        steps=[
            ("features", make_feature_transformer()),
            ("preprocess", make_preprocessor(sample_features)),
            ("model", make_regressor(model_name=model_name, random_state=random_state)),
        ]
    )
