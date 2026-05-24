"""Dataset preparation helpers for forecasting."""

from __future__ import annotations

import pandas as pd

from app.forecasting.features import MODEL_FEATURE_COLUMNS, build_zone_feature_frame, normalize_zone_id


def build_training_dataset(
    sensor_frame: pd.DataFrame,
    zone_id: str,
) -> pd.DataFrame:
    normalized_zone_id = normalize_zone_id(zone_id)
    feature_frame = build_zone_feature_frame(sensor_frame, zone_id=normalized_zone_id, dropna_lags=True)
    dataset = feature_frame.copy()
    dataset["target_load_mw"] = dataset["load_mw"].shift(-1)
    dataset = dataset.dropna(subset=["target_load_mw"]).reset_index(drop=True)
    return dataset


def split_train_test(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = max(24, int(len(dataset) * 0.8))
    if split_index >= len(dataset):
        split_index = len(dataset) - 1
    train_frame = dataset.iloc[:split_index].copy()
    test_frame = dataset.iloc[split_index:].copy()
    return train_frame, test_frame


def extract_model_matrices(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return dataset.loc[:, MODEL_FEATURE_COLUMNS], dataset["target_load_mw"]
