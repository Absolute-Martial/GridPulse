import pandas as pd
import numpy as np

from app.forecasting.schema import CANONICAL_HISTORY_COLUMNS, CANONICAL_RESOLUTION_MINUTES
from app.simulator.ami_history_generator import (
    _ambient_temperature,
    _resolve_season,
    generate_ami_history,
)


def test_resolve_season_uses_nepal_style_five_season_mapping() -> None:
    assert _resolve_season(12) == ("winter", 0)
    assert _resolve_season(3) == ("spring", 1)
    assert _resolve_season(5) == ("summer", 2)
    assert _resolve_season(7) == ("monsoon", 3)
    assert _resolve_season(10) == ("autumn", 4)


def test_ambient_temperature_supports_autumn_season_index() -> None:
    rng = np.random.default_rng(7)
    value = _ambient_temperature(pd.Timestamp("2026-10-15T12:00:00Z"), 4, rng)

    assert isinstance(value, float)


def test_generate_ami_history_returns_canonical_15_minute_schema() -> None:
    frame = generate_ami_history(days=7, seed=7)

    assert not frame.empty
    assert list(frame.columns) == list(CANONICAL_HISTORY_COLUMNS)
    assert set(frame["entity_type"]) >= {
        "substation",
        "feeder",
        "customer_group",
        "enterprise",
    }
    assert frame["timestamp"].nunique() >= 7 * 24 * 4


def test_generate_ami_history_covers_15_minute_intervals_for_each_timestamp() -> None:
    frame = generate_ami_history(days=7, seed=7)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True).drop_duplicates().sort_values()

    assert timestamps.iloc[0].minute == 0
    assert timestamps.iloc[-1].minute == 45
    deltas = timestamps.diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(minutes=CANONICAL_RESOLUTION_MINUTES)
