from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from typing import Callable

from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import ID_COLUMN


COUNTRY_NORMALIZATION = {
    "Spain": "Ispanya",
    "Sweden": "Isvec",
    "South Korea": "Guney Kore",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series | float) -> pd.Series:
    return numerator / (denominator + 1e-3)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features from sleep, stress, activity, and health fields."""
    out = df.copy()

    rem = out["rem_yuzdesi"]
    deep = out["derin_uyku_yuzdesi"]
    latency = out["uykuya_dalma_suresi_dk"]
    wakeups = out["gecelik_uyanma_sayisi"]
    caffeine = out["uyku_oncesi_kafein_mg"]
    screen = out["uyku_oncesi_ekran_suresi_dk"]
    steps = out["gunluk_adim_sayisi"]
    nap = out["sekerleme_suresi_dk"]
    stress = out["stres_skoru"]
    work = out["gunluk_calisma_saati"]
    bmi = out["vucut_kitle_indeksi"]
    age = out["yas"]
    resting_hr = out["dinlenik_nabiz_bpm"]
    room_temp = out["oda_sicakligi_celsius"]
    weekend_diff = out["hafta_sonu_uyku_farki_saat"]

    out["ulke_norm"] = out["ulke"].replace(COUNTRY_NORMALIZATION)
    out["toplam_rem_derin_uyku_yuzdesi"] = rem + deep
    out["rem_derin_orani"] = _safe_divide(rem, deep)
    out["derin_rem_orani"] = _safe_divide(deep, rem)
    out["uyku_kalitesi_proxy"] = rem + deep - 0.18 * latency - 1.15 * wakeups
    out["uyku_bolunme_yuku"] = wakeups * latency
    out["uyku_bolunme_skoru"] = wakeups + latency / 30.0
    out["uyku_oncesi_uyarici_yuk"] = screen / 60.0 + caffeine / 100.0
    out["kafein_ekran_carpim"] = caffeine * screen
    out["adim_bin"] = steps / 1000.0
    out["aktivite_calisma_orani"] = steps / (work + 1.0)
    out["adim_bin_per_saat"] = out["adim_bin"] / (work + 1.0)
    out["stres_calisma_carpim"] = stress * work
    out["stres_uyku_bolunme_carpim"] = stress * (wakeups + 1.0)
    out["stres_kalitesi_orani"] = stress / (rem + deep + 1.0)
    out["dinlenme_yuku"] = stress + work + wakeups
    out["kardiyo_yuk"] = resting_hr + 0.35 * bmi - steps / 5000.0
    out["ideal_sicaklik_farki"] = (room_temp - 19.0).abs()
    out["hafta_sonu_fark_abs"] = weekend_diff.abs()
    out["hafta_sonu_fark_pozitif"] = weekend_diff.clip(lower=0)
    out["yas_kare"] = age**2
    out["bmi_kare"] = bmi**2
    out["sekerleme_saat"] = nap / 60.0
    out["sekerleme_uzun_mu"] = nap.gt(30).astype(float)

    gun_tipi = out["gun_tipi"].astype("string")
    kronotip = out["kronotip"].astype("string")
    ruh = out["ruh_sagligi_durumu"].astype("string")
    mevsim = out["mevsim"].astype("string")
    meslek = out["meslek"].astype("string")

    out["hafta_sonu_mu"] = gun_tipi.eq("Hafta sonu").fillna(False).astype(float)
    out["saglikli_mi"] = ruh.eq("Saglikli").fillna(False).astype(float)
    out["gece_insani_mi"] = kronotip.eq("Gece insani").fillna(False).astype(float)
    out["hafta_sonu_kronotip"] = gun_tipi.fillna("Missing") + "__" + kronotip.fillna("Missing")
    out["mevsim_gun_tipi"] = mevsim.fillna("Missing") + "__" + gun_tipi.fillna("Missing")
    out["ruh_kronotip"] = ruh.fillna("Missing") + "__" + kronotip.fillna("Missing")
    out["meslek_gun_tipi"] = meslek.fillna("Missing") + "__" + gun_tipi.fillna("Missing")

    out["yas_grubu"] = pd.cut(
        age,
        bins=[17, 24, 34, 44, 54, np.inf],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    ).astype("object")
    out["bmi_sinifi"] = pd.cut(
        bmi,
        bins=[-np.inf, 18.5, 25.0, 30.0, 35.0, np.inf],
        labels=["dusuk", "normal", "fazla", "obez_1", "obez_2_plus"],
    ).astype("object")
    out["stres_grubu"] = pd.cut(
        stress,
        bins=[-np.inf, 3.0, 6.0, 8.0, np.inf],
        labels=["dusuk", "orta", "yuksek", "cok_yuksek"],
    ).astype("object")

    return out


def add_improved_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the stronger feature set selected by cross-validation experiments."""
    out = add_features(df)

    rem = out["rem_yuzdesi"]
    deep = out["derin_uyku_yuzdesi"]
    latency = out["uykuya_dalma_suresi_dk"]
    wakeups = out["gecelik_uyanma_sayisi"]
    caffeine = out["uyku_oncesi_kafein_mg"]
    screen = out["uyku_oncesi_ekran_suresi_dk"]
    steps = out["gunluk_adim_sayisi"]
    nap = out["sekerleme_suresi_dk"]
    stress = out["stres_skoru"]
    work = out["gunluk_calisma_saati"]
    bmi = out["vucut_kitle_indeksi"]
    resting_hr = out["dinlenik_nabiz_bpm"]
    room_temp = out["oda_sicakligi_celsius"]

    out["isik_uyku_yuzdesi"] = 100.0 - rem - deep
    out["sleep_quality_ratio_v2"] = deep / (rem + 1.0)
    out["sleep_fragmentation_v2"] = wakeups * latency
    out["pre_sleep_stimulation"] = caffeine + screen
    out["screen_caffeine_interaction_v2"] = caffeine * screen
    out["activity_stress_ratio"] = steps / (stress + 1.0)
    out["work_stress_interaction_v2"] = work * stress
    out["bmi_resting_hr_interaction"] = bmi * resting_hr
    out["weekend_sleep_instability_v2"] = out["hafta_sonu_uyku_farki_saat"].abs()
    out["nap_activity_ratio"] = nap / (steps + 1.0)
    out["temp_comfort_quadratic"] = (room_temp - 19.0) ** 2
    out["sleep_efficiency_proxy"] = (
        rem
        + deep
        - wakeups * 1.25
        - latency * 0.20
        - screen / 120.0
        - caffeine / 200.0
    )
    out["recovery_load_balance"] = (rem + deep + steps / 1000.0) / (
        stress + work + wakeups + 1.0
    )
    out["mental_stress_load"] = stress * (1.0 - out["saglikli_mi"])
    out["weekend_chronotype_stress"] = (
        out["hafta_sonu_mu"] * stress * (out["gece_insani_mi"] + 1.0)
    )

    stress_group = out["stres_grubu"].astype("string").fillna("Missing")
    out["meslek_stres_grubu"] = (
        out["meslek"].astype("string").fillna("Missing") + "__" + stress_group
    )
    out["ruh_stres_grubu"] = (
        out["ruh_sagligi_durumu"].astype("string").fillna("Missing") + "__" + stress_group
    )
    out["ulke_gun_tipi"] = (
        out["ulke_norm"].astype("string").fillna("Missing")
        + "__"
        + out["gun_tipi"].astype("string").fillna("Missing")
    )

    return out


def infer_feature_columns(
    df: pd.DataFrame,
    feature_func: Callable[[pd.DataFrame], pd.DataFrame] = add_features,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns after feature engineering."""
    engineered = feature_func(df)
    numeric_columns = [
        col for col in engineered.select_dtypes(include=[np.number]).columns if col != ID_COLUMN
    ]
    categorical_columns = [
        col for col in engineered.columns if col not in numeric_columns and col != ID_COLUMN
    ]
    return numeric_columns, categorical_columns


def make_preprocessor(
    df: pd.DataFrame,
    feature_func: Callable[[pd.DataFrame], pd.DataFrame] = add_features,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build the preprocessing transformer from a representative raw feature frame."""
    numeric_columns, categorical_columns = infer_feature_columns(df, feature_func=feature_func)

    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", min_frequency=5, sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", categorical_transformer, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_feature_transformer(
    feature_func: Callable[[pd.DataFrame], pd.DataFrame] = add_features,
) -> FunctionTransformer:
    return FunctionTransformer(feature_func, validate=False)


def _to_native_hgb_frame(df: pd.DataFrame, feature_func: Callable[[pd.DataFrame], pd.DataFrame]):
    out = feature_func(df).drop(columns=[ID_COLUMN])
    for col in out.select_dtypes(exclude="number").columns:
        out[col] = out[col].astype("category")
    return out


def to_native_hgb_current_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return current features with pandas categorical dtypes for native HGB splits."""
    return _to_native_hgb_frame(df, add_features)


def to_native_hgb_improved_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return improved features with pandas categorical dtypes for native HGB splits."""
    return _to_native_hgb_frame(df, add_improved_features)
