import pandas as pd

from app.forecasting.feeder_fingerprint import (
    FINGERPRINT_KEY_COLUMNS,
    FINGERPRINT_VALUE_COLUMNS,
    build_fingerprint_database,
    merge_fingerprint_features,
)
from app.simulator.ami_history_generator import generate_ami_history


def test_build_fingerprint_database_returns_quantiles() -> None:
    history = generate_ami_history(days=14, seed=7)
    fingerprint = build_fingerprint_database(history)

    assert not fingerprint.empty
    assert {*FINGERPRINT_KEY_COLUMNS, *FINGERPRINT_VALUE_COLUMNS} <= set(fingerprint.columns)
    assert fingerprint["fingerprint_p10_kw"].le(fingerprint["fingerprint_mean_kw"]).all()
    assert fingerprint["fingerprint_mean_kw"].le(fingerprint["fingerprint_p90_kw"]).all()


def test_merge_fingerprint_features_restores_lookup_values() -> None:
    history = generate_ami_history(days=14, seed=7)
    fingerprint = build_fingerprint_database(history)

    merged = merge_fingerprint_features(history, fingerprint)

    assert len(merged) == len(history)
    assert list(FINGERPRINT_VALUE_COLUMNS) == [
        "fingerprint_mean_kw",
        "fingerprint_p10_kw",
        "fingerprint_p90_kw",
    ]
    assert merged.loc[:, list(FINGERPRINT_VALUE_COLUMNS)].notna().all().all()
