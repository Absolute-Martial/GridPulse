"""Fingerprint database helpers for 15-minute forecasting history."""

from __future__ import annotations

from typing import Final

import pandas as pd

from app.forecasting.schema import ensure_canonical_history

FINGERPRINT_KEY_COLUMNS: Final[tuple[str, ...]] = (
    "entity_type",
    "entity_id",
    "day_of_week",
    "slot_index",
)

FINGERPRINT_VALUE_COLUMNS: Final[tuple[str, ...]] = (
    "fingerprint_mean_kw",
    "fingerprint_p10_kw",
    "fingerprint_p90_kw",
)


def build_fingerprint_database(history: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic 15-minute fingerprint quantiles from canonical history."""

    frame = ensure_canonical_history(history)
    fingerprint = (
        frame.groupby(list(FINGERPRINT_KEY_COLUMNS), as_index=False)
        .agg(
            fingerprint_mean_kw=("load_kw", "mean"),
            fingerprint_p10_kw=("load_kw", lambda series: float(series.quantile(0.10))),
            fingerprint_p90_kw=("load_kw", lambda series: float(series.quantile(0.90))),
        )
        .sort_values(list(FINGERPRINT_KEY_COLUMNS))
        .reset_index(drop=True)
    )
    return fingerprint


def merge_fingerprint_features(
    history: pd.DataFrame,
    fingerprint_database: pd.DataFrame,
) -> pd.DataFrame:
    """Attach fingerprint reference columns back onto canonical history rows."""

    frame = ensure_canonical_history(history).drop(columns=list(FINGERPRINT_VALUE_COLUMNS))
    merged = frame.merge(
        fingerprint_database.loc[:, [*FINGERPRINT_KEY_COLUMNS, *FINGERPRINT_VALUE_COLUMNS]],
        on=list(FINGERPRINT_KEY_COLUMNS),
        how="left",
        validate="many_to_one",
    )
    return merged

