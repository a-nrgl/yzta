from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from src.config import RANDOM_STATE
from src.features import (
    add_features,
    add_improved_features,
    make_feature_transformer,
    make_preprocessor,
    to_native_hgb_current_frame,
    to_native_hgb_improved_frame,
)


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
        "Use one of: ridge, hgb, hgb_regularized, extra_trees, ensemble, improved_ensemble."
    )


def _make_hgb(random_state: int = RANDOM_STATE, **kwargs) -> HistGradientBoostingRegressor:
    params = {
        "loss": "squared_error",
        "early_stopping": True,
        "validation_fraction": 0.12,
        "random_state": random_state,
    }
    params.update(kwargs)
    return HistGradientBoostingRegressor(**params)


def _make_ohe_pipeline(
    sample_features: pd.DataFrame,
    estimator,
    *,
    feature_func=add_features,
    scale_numeric: bool = True,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("features", make_feature_transformer(feature_func)),
            (
                "preprocess",
                make_preprocessor(
                    sample_features,
                    feature_func=feature_func,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", estimator),
        ]
    )


def _make_native_hgb_pipeline(frame_transformer, estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("features_for_hgb", FunctionTransformer(frame_transformer, validate=False)),
            ("model", estimator),
        ]
    )


def build_improved_ensemble_pipeline(
    sample_features: pd.DataFrame,
    random_state: int = RANDOM_STATE,
) -> VotingRegressor:
    """Build the CV-selected weighted ensemble used for the improved submission."""
    ridge_plus = _make_ohe_pipeline(
        sample_features,
        RidgeCV(alphas=np.logspace(-2, 4, 25)),
        feature_func=add_improved_features,
        scale_numeric=True,
    )
    hgb_main_plus = _make_ohe_pipeline(
        sample_features,
        _make_hgb(
            random_state=random_state,
            max_iter=1100,
            learning_rate=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            l2_regularization=0.05,
        ),
        feature_func=add_improved_features,
        scale_numeric=False,
    )
    hgb_regularized_plus = _make_ohe_pipeline(
        sample_features,
        _make_hgb(
            random_state=random_state + 17,
            max_iter=800,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=35,
            l2_regularization=0.08,
        ),
        feature_func=add_improved_features,
        scale_numeric=False,
    )
    hgb_slow_plus = _make_ohe_pipeline(
        sample_features,
        _make_hgb(
            random_state=random_state + 29,
            max_iter=1400,
            learning_rate=0.022,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.12,
        ),
        feature_func=add_improved_features,
        scale_numeric=False,
    )
    hgb_native_current = _make_native_hgb_pipeline(
        to_native_hgb_current_frame,
        _make_hgb(
            random_state=random_state + 11,
            categorical_features="from_dtype",
            max_iter=900,
            learning_rate=0.04,
            max_leaf_nodes=15,
            min_samples_leaf=35,
            l2_regularization=0.08,
        ),
    )
    hgb_native_small_leaf = _make_native_hgb_pipeline(
        to_native_hgb_improved_frame,
        _make_hgb(
            random_state=random_state + 101,
            categorical_features="from_dtype",
            max_iter=700,
            learning_rate=0.06,
            max_leaf_nodes=8,
            min_samples_leaf=45,
            l2_regularization=0.10,
        ),
    )
    hgb_native_slow = _make_native_hgb_pipeline(
        to_native_hgb_improved_frame,
        _make_hgb(
            random_state=random_state + 105,
            categorical_features="from_dtype",
            max_iter=1500,
            learning_rate=0.025,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=0.15,
        ),
    )

    return VotingRegressor(
        estimators=[
            ("ridge_plus", ridge_plus),
            ("hgb_main_plus", hgb_main_plus),
            ("hgb_regularized_plus", hgb_regularized_plus),
            ("hgb_slow_plus", hgb_slow_plus),
            ("hgb_native_current", hgb_native_current),
            ("hgb_native_small_leaf", hgb_native_small_leaf),
            ("hgb_native_slow", hgb_native_slow),
        ],
        weights=[0.41, 0.02, 0.10, 0.04, 0.19, 0.17, 0.07],
        n_jobs=1,
    )


def build_model_pipeline(
    sample_features: pd.DataFrame,
    model_name: str = "ensemble",
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build the full feature engineering, preprocessing, and model pipeline."""
    if model_name.lower() == "improved_ensemble":
        return build_improved_ensemble_pipeline(
            sample_features=sample_features,
            random_state=random_state,
        )

    return Pipeline(
        steps=[
            ("features", make_feature_transformer()),
            ("preprocess", make_preprocessor(sample_features)),
            ("model", make_regressor(model_name=model_name, random_state=random_state)),
        ]
    )
