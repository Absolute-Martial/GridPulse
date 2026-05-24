# Secondary Substation Forecasting MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline forecasting-only GridPulse MVP around canonical 15-minute AMI/SMU history, with fingerprint baseline, tree-based forecasting, rolling inference, structured errors, and tree-model explanation.

**Architecture:** One canonical 15-minute forecasting stack handles substation, feeder, customer-group, and enterprise targets through shared feature engineering and a pluggable model registry. All forecast horizons return slot-level predictions, with optional derived aggregates, while anomaly, GNN, OPF, routing, recommendation, allocation, and control modules remain inactive.

**Tech Stack:** FastAPI, pandas, NumPy, scikit-learn, SHAP with permutation fallback, optional PyTorch for TCN/N-HiTS, static Docker-served frontend, pytest, httpx

---

## File Structure

### Active backend files

- Create: `backend/app/forecasting/schema.py`
  - Validate the canonical 15-minute AMI schema, normalize `horizon` strings, derive `load_kw` from `interval_energy_kwh` when needed, and enforce structured error categories.
- Create: `backend/app/simulator/ami_history_generator.py`
  - Generate 15-minute AMI/SMU training history for substation, feeder, customer-group, and enterprise targets.
- Create: `backend/app/forecasting/feeder_fingerprint.py`
  - Build fingerprint tables at 15-minute resolution and expose lookup helpers.
- Modify: `backend/app/forecasting/base.py`
  - Replace zone-specific signatures with target-aware forecasting interfaces.
- Modify: `backend/app/forecasting/features.py`
  - Move from hourly zone features to 15-minute target-aware rolling windows and leakage-safe features.
- Create: `backend/app/forecasting/fingerprint_baseline.py`
  - Implement the fallback baseline model using fingerprint profiles.
- Create: `backend/app/forecasting/tree_forecaster.py`
  - Implement the offline tree-based forecasting model with residual-based confidence bands.
- Modify: `backend/app/forecasting/metrics.py`
  - Add peak time error and peak load error.
- Modify: `backend/app/forecasting/model_registry.py`
  - Register `fingerprint_baseline`, `tree`, `tcn`, and `nhits`, and remove old zone-only assumptions.
- Modify: `backend/app/forecasting/tcn.py`
  - Adapt the optional TCN to the new target-aware 15-minute interface.
- Create: `backend/app/forecasting/nhits.py`
  - Provide the N-HiTS file under the requested name while preserving optional behavior.
- Create: `backend/app/forecasting/response_formatter.py`
  - Produce the canonical `slot_predictions`, `aggregated_summary`, and `summary` payloads.
- Create: `backend/app/explainability/shap_forecast_explainer.py`
  - Explain peak-focused tree forecasts via SHAP or permutation fallback.
- Modify: `backend/app/api/forecasting.py`
  - Replace current zone-based endpoints with target-aware forecasting APIs.
- Create: `backend/app/api/explainability.py`
  - Add the explanation endpoint surface.
- Modify: `backend/app/api/router.py`
  - Wire the new forecasting and explainability routers without invoking inactive modules.
- Modify: `backend/app/core/config.py`
  - Add offline paths for AMI history, fingerprints, forecasting models, and explanation artifacts.

### Active frontend files

- Modify: `frontend/app.js`
  - Replace the old broad dashboard forecasting view with forecasting-only API integrations.
- Modify: `frontend/styles.css`
  - Add minimal styles for forecasting result tables and explanation cards.
- Modify: `frontend/index.html`
  - Keep the existing static app shell, but change labels and controls to the forecasting-only workflow.

### Tests

- Create: `backend/tests/test_ami_history_generator.py`
- Create: `backend/tests/test_feeder_fingerprint.py`
- Create: `backend/tests/test_forecasting_features.py`
- Create: `backend/tests/test_tree_forecaster.py`
- Create: `backend/tests/test_response_formatter.py`
- Create: `backend/tests/test_forecast_explainability.py`
- Modify: `backend/tests/test_forecasting.py`
  - Replace zone-based assertions with target-aware 15-minute forecasting API tests.
- Keep inactive-module tests untouched unless they break due to shared imports.

## Task 1: Add Schema Validation and Horizon Normalization

**Files:**
- Create: `backend/app/forecasting/schema.py`
- Modify: `backend/app/core/config.py`
- Test: `backend/tests/test_forecasting_features.py`

- [ ] **Step 1: Write the failing schema and horizon tests**

```python
import pandas as pd
import pytest

from app.forecasting.schema import (
    ForecastEntityError,
    ForecastHistoryError,
    ForecastHorizonError,
    ensure_canonical_history,
    normalize_horizon,
)


def test_normalize_horizon_accepts_canonical_strings() -> None:
    assert normalize_horizon("1h") == ("1h", 4)
    assert normalize_horizon("4h") == ("4h", 16)
    assert normalize_horizon("24h") == ("24h", 96)


def test_normalize_horizon_rejects_invalid_values() -> None:
    with pytest.raises(ForecastHorizonError):
        normalize_horizon("6")


def test_ensure_canonical_history_derives_load_kw_from_interval_energy() -> None:
    frame = pd.DataFrame(
        [
            {
                "timestamp": "2026-05-25T10:15:00Z",
                "entity_type": "feeder",
                "entity_id": "F_RES_01",
                "secondary_substation_id": "SS_01",
                "transformer_id": "TR_01",
                "feeder_id": "F_RES_01",
                "feeder_type": "residential",
                "customer_group_id": "CG_01",
                "customer_type": "residential",
                "enterprise_id": "",
                "is_dedicated_line": 0,
                "contracted_md_kw": 250.0,
                "interval_energy_kwh": 36.0,
                "temperature_c": 27.0,
                "humidity_percent": 61.0,
                "day_type": "weekday",
                "hour": 10,
                "minute": 15,
                "slot_index": 41,
                "month": 5,
                "day_of_week": 0,
                "season": "summer",
                "season_index": 2,
                "is_weekend": 0,
                "is_holiday": 0,
                "data_quality_flag": "ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": 140.0,
                "fingerprint_p10_kw": 128.0,
                "fingerprint_p90_kw": 154.0,
            }
        ]
    )

    validated = ensure_canonical_history(frame)
    assert validated.loc[0, "load_kw"] == 144.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecasting_features.py -k "horizon or canonical"`

Expected: FAIL with missing `schema.py` helpers.

- [ ] **Step 3: Implement the schema module and config paths**

```python
from dataclasses import dataclass

CANONICAL_HISTORY_COLUMNS = (
    "timestamp",
    "entity_type",
    "entity_id",
    "secondary_substation_id",
    "transformer_id",
    "feeder_id",
    "feeder_type",
    "customer_group_id",
    "customer_type",
    "enterprise_id",
    "is_dedicated_line",
    "contracted_md_kw",
    "load_kw",
    "interval_energy_kwh",
    "temperature_c",
    "humidity_percent",
    "day_type",
    "hour",
    "minute",
    "slot_index",
    "month",
    "day_of_week",
    "season",
    "season_index",
    "is_weekend",
    "is_holiday",
    "data_quality_flag",
    "source_type",
    "production_schedule_kw",
    "fingerprint_mean_kw",
    "fingerprint_p10_kw",
    "fingerprint_p90_kw",
)

HORIZON_TO_STEPS = {"1h": 4, "4h": 16, "24h": 96}


def normalize_horizon(horizon: str) -> tuple[str, int]:
    normalized = horizon.strip().lower()
    if normalized not in HORIZON_TO_STEPS:
        raise ForecastHorizonError(f"Invalid horizon '{horizon}'. Expected one of {sorted(HORIZON_TO_STEPS)}.")
    return normalized, HORIZON_TO_STEPS[normalized]
```

- [ ] **Step 4: Run the schema tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecasting_features.py -k "horizon or canonical"`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/schema.py \
  backend/app/core/config.py \
  backend/tests/test_forecasting_features.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add canonical forecasting schema validation"
```

## Task 2: Build the Synthetic 15-Minute AMI History Generator

**Files:**
- Create: `backend/app/simulator/ami_history_generator.py`
- Modify: `backend/app/core/config.py`
- Test: `backend/tests/test_ami_history_generator.py`

- [ ] **Step 1: Write the failing generator tests**

```python
from app.simulator.ami_history_generator import generate_ami_history


def test_generate_ami_history_returns_15_minute_schema() -> None:
    frame = generate_ami_history(days=7, seed=7)

    assert not frame.empty
    assert {"entity_type", "entity_id", "load_kw", "interval_energy_kwh", "season", "slot_index"} <= set(frame.columns)
    assert set(frame["entity_type"]) >= {"substation", "feeder", "customer_group", "enterprise"}
    assert frame["timestamp"].nunique() >= 7 * 24 * 4
```

- [ ] **Step 2: Run the generator test to verify it fails**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_ami_history_generator.py`

Expected: FAIL because `ami_history_generator.py` does not exist.

- [ ] **Step 3: Implement the AMI generator**

```python
def generate_ami_history(days: int = 30, seed: int | None = None) -> pd.DataFrame:
    timestamps = pd.date_range(end=pd.Timestamp.utcnow().floor("15min"), periods=days * 24 * 4, freq="15min", tz="UTC")
    rows: list[dict[str, object]] = []
    for timestamp in timestamps:
        season, season_index = resolve_season(timestamp.month)
        for feeder in FEEDER_CATALOG:
            load_kw = simulate_feeder_load(feeder, timestamp, season_index)
            rows.append(
                build_ami_row(
                    timestamp=timestamp,
                    feeder=feeder,
                    load_kw=load_kw,
                    interval_energy_kwh=load_kw * 0.25,
                    season=season,
                    season_index=season_index,
                )
            )
    frame = pd.DataFrame(rows)
    return ensure_canonical_history(frame)
```

- [ ] **Step 4: Run the generator tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_ami_history_generator.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/simulator/ami_history_generator.py \
  backend/tests/test_ami_history_generator.py \
  backend/app/core/config.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add synthetic 15-minute AMI history generator"
```

## Task 3: Build the Fingerprint Database Layer

**Files:**
- Create: `backend/app/forecasting/feeder_fingerprint.py`
- Test: `backend/tests/test_feeder_fingerprint.py`

- [ ] **Step 1: Write the failing fingerprint tests**

```python
from app.forecasting.feeder_fingerprint import build_fingerprint_database
from app.simulator.ami_history_generator import generate_ami_history


def test_build_fingerprint_database_returns_quantiles() -> None:
    frame = generate_ami_history(days=14, seed=7)
    fingerprint = build_fingerprint_database(frame)

    assert not fingerprint.empty
    assert {"entity_type", "entity_id", "slot_index", "fingerprint_mean_kw", "fingerprint_p10_kw", "fingerprint_p90_kw"} <= set(fingerprint.columns)
```

- [ ] **Step 2: Run the fingerprint test to verify it fails**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_feeder_fingerprint.py`

Expected: FAIL because the fingerprint builder does not exist.

- [ ] **Step 3: Implement the fingerprint builder**

```python
def build_fingerprint_database(history: pd.DataFrame) -> pd.DataFrame:
    frame = ensure_canonical_history(history)
    grouped = (
        frame.groupby(["entity_type", "entity_id", "day_of_week", "slot_index"], as_index=False)
        .agg(
            fingerprint_mean_kw=("load_kw", "mean"),
            fingerprint_p10_kw=("load_kw", lambda s: float(s.quantile(0.10))),
            fingerprint_p90_kw=("load_kw", lambda s: float(s.quantile(0.90))),
        )
    )
    return grouped.sort_values(["entity_type", "entity_id", "day_of_week", "slot_index"]).reset_index(drop=True)
```

- [ ] **Step 4: Run the fingerprint tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_feeder_fingerprint.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/feeder_fingerprint.py \
  backend/tests/test_feeder_fingerprint.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add forecasting fingerprint database"
```

## Task 4: Replace Hourly Zone Features with 15-Minute Target Features

**Files:**
- Modify: `backend/app/forecasting/features.py`
- Create: `backend/tests/test_forecasting_features.py`

- [ ] **Step 1: Write the failing feature-engineering tests**

```python
from app.forecasting.features import build_target_feature_frame
from app.simulator.ami_history_generator import generate_ami_history


def test_build_target_feature_frame_respects_lookback_and_no_future_leakage() -> None:
    history = generate_ami_history(days=7, seed=7)
    feature_frame = build_target_feature_frame(
        history,
        entity_type="feeder",
        entity_id="F_RES_01",
        lookback_steps=480,
    )

    assert len(feature_frame) >= 96
    assert {"lag_1", "lag_4", "lag_96", "rolling_mean_4", "rolling_mean_16", "rolling_mean_96"} <= set(feature_frame.columns)
    assert feature_frame["target_load_kw"].isna().sum() == 0
```

- [ ] **Step 2: Run the feature tests to verify they fail**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecasting_features.py`

Expected: FAIL because the old zone-hour feature API no longer matches.

- [ ] **Step 3: Implement the 15-minute target feature builder**

```python
MODEL_FEATURE_COLUMNS = (
    "contracted_md_kw",
    "temperature_c",
    "humidity_percent",
    "hour",
    "minute",
    "slot_index",
    "day_of_week",
    "month",
    "season_index",
    "is_weekend",
    "is_holiday",
    "production_schedule_kw",
    "fingerprint_mean_kw",
    "fingerprint_p10_kw",
    "fingerprint_p90_kw",
    "lag_1",
    "lag_4",
    "lag_16",
    "lag_96",
    "rolling_mean_4",
    "rolling_mean_16",
    "rolling_mean_96",
)


def build_target_feature_frame(history: pd.DataFrame, entity_type: str, entity_id: str, lookback_steps: int = 480) -> pd.DataFrame:
    target = filter_target_history(history, entity_type=entity_type, entity_id=entity_id).sort_values("timestamp").reset_index(drop=True)
    target["lag_1"] = target["load_kw"].shift(1)
    target["lag_4"] = target["load_kw"].shift(4)
    target["lag_16"] = target["load_kw"].shift(16)
    target["lag_96"] = target["load_kw"].shift(96)
    shifted = target["load_kw"].shift(1)
    target["rolling_mean_4"] = shifted.rolling(4, min_periods=1).mean()
    target["rolling_mean_16"] = shifted.rolling(16, min_periods=1).mean()
    target["rolling_mean_96"] = shifted.rolling(96, min_periods=1).mean()
    return target.iloc[-lookback_steps:].copy()
```

- [ ] **Step 4: Run the feature tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecasting_features.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/features.py \
  backend/tests/test_forecasting_features.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add 15-minute target feature engineering"
```

## Task 5: Add the Fingerprint Baseline Model

**Files:**
- Create: `backend/app/forecasting/fingerprint_baseline.py`
- Modify: `backend/app/forecasting/base.py`
- Test: `backend/tests/test_tree_forecaster.py`

- [ ] **Step 1: Write the failing baseline-model test**

```python
from app.forecasting.fingerprint_baseline import FingerprintBaselineForecaster
from app.forecasting.feeder_fingerprint import build_fingerprint_database
from app.simulator.ami_history_generator import generate_ami_history


def test_fingerprint_baseline_predicts_slot_sequence() -> None:
    history = generate_ami_history(days=14, seed=7)
    fingerprint = build_fingerprint_database(history)
    model = FingerprintBaselineForecaster()
    model.fit_fingerprint(fingerprint)
    prediction = model.predict(history, entity_type="feeder", entity_id="F_RES_01", horizon="1h")

    assert prediction["horizon"] == "1h"
    assert len(prediction["slot_predictions"]) == 4
```

- [ ] **Step 2: Run the baseline test to verify it fails**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tree_forecaster.py -k fingerprint`

Expected: FAIL because the baseline forecaster does not exist yet.

- [ ] **Step 3: Implement the baseline forecaster and target-aware base interface**

```python
class BaseForecaster(ABC):
    @abstractmethod
    def train(self, dataframe: pd.DataFrame, entity_type: str, entity_id: str, horizon: str) -> dict[str, Any]:
        ...

    @abstractmethod
    def predict(self, dataframe: pd.DataFrame, entity_type: str, entity_id: str, horizon: str) -> dict[str, Any]:
        ...


class FingerprintBaselineForecaster(BaseForecaster):
    model_name = "fingerprint_baseline"

    def predict(self, dataframe: pd.DataFrame, entity_type: str, entity_id: str, horizon: str) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        anchor = get_latest_target_timestamp(dataframe, entity_type=entity_type, entity_id=entity_id)
        return build_baseline_prediction_payload(...)
```

- [ ] **Step 4: Run the baseline test again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tree_forecaster.py -k fingerprint`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/base.py \
  backend/app/forecasting/fingerprint_baseline.py \
  backend/tests/test_tree_forecaster.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add fingerprint baseline forecaster"
```

## Task 6: Add the Tree-Based Forecasting Model

**Files:**
- Create: `backend/app/forecasting/tree_forecaster.py`
- Modify: `backend/app/forecasting/model_registry.py`
- Test: `backend/tests/test_tree_forecaster.py`

- [ ] **Step 1: Write the failing tree forecaster tests**

```python
from app.forecasting.tree_forecaster import TreeForecaster
from app.simulator.ami_history_generator import generate_ami_history


def test_tree_forecaster_trains_and_predicts_target_slots(tmp_path) -> None:
    history = generate_ami_history(days=21, seed=7)
    forecaster = TreeForecaster(model_dir=tmp_path)
    train_result = forecaster.train(history, entity_type="feeder", entity_id="F_RES_01", horizon="4h")
    prediction = forecaster.predict(history, entity_type="feeder", entity_id="F_RES_01", horizon="4h")

    assert train_result["artifact_path"].endswith("tree_feeder_F_RES_01_4h.pkl")
    assert len(prediction["slot_predictions"]) == 16
    assert prediction["summary"]["confidence"] in {"high", "medium", "low"}
```

- [ ] **Step 2: Run the tree forecaster tests to verify they fail**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tree_forecaster.py -k tree`

Expected: FAIL because `tree_forecaster.py` does not exist.

- [ ] **Step 3: Implement the tree-based model**

```python
class TreeForecaster(BaseForecaster):
    model_name = "tree"

    def train(self, dataframe: pd.DataFrame, entity_type: str, entity_id: str, horizon: str) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        dataset = build_supervised_dataset(dataframe, entity_type=entity_type, entity_id=entity_id, horizon_steps=horizon_steps)
        estimator = RandomForestRegressor(
            n_estimators=300,
            random_state=7,
            min_samples_leaf=2,
        )
        estimator.fit(dataset.feature_matrix, dataset.target_matrix)
        self.artifact = {
            "model_name": self.model_name,
            "entity_type": entity_type,
            "entity_id": entity_id,
            "horizon": normalized_horizon,
            "model": estimator,
            "residual_quantiles": calibrate_residual_quantiles(dataset, estimator),
        }
        self.save(build_model_path(...))
        return {...}
```

- [ ] **Step 4: Run the tree forecaster tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tree_forecaster.py -k tree`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/tree_forecaster.py \
  backend/app/forecasting/model_registry.py \
  backend/tests/test_tree_forecaster.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add tree-based forecasting model"
```

## Task 7: Add the Forecast Response Formatter and Structured Errors

**Files:**
- Create: `backend/app/forecasting/response_formatter.py`
- Test: `backend/tests/test_response_formatter.py`

- [ ] **Step 1: Write the failing response-format test**

```python
from app.forecasting.response_formatter import build_forecast_response


def test_build_forecast_response_uses_aggregated_summary_and_energy_rule() -> None:
    payload = build_forecast_response(
        entity_type="feeder",
        entity_id="F_RES_01",
        horizon="1h",
        latest_timestamp="2026-05-25T10:15:00Z",
        slot_predictions=[
            {"timestamp": "2026-05-25T10:30:00Z", "predicted_load_kw": 100.0, "p10_kw": 90.0, "p90_kw": 110.0, "fingerprint_mean_kw": 98.0},
            {"timestamp": "2026-05-25T10:45:00Z", "predicted_load_kw": 120.0, "p10_kw": 100.0, "p90_kw": 130.0, "fingerprint_mean_kw": 118.0},
            {"timestamp": "2026-05-25T11:00:00Z", "predicted_load_kw": 140.0, "p10_kw": 120.0, "p90_kw": 150.0, "fingerprint_mean_kw": 138.0},
            {"timestamp": "2026-05-25T11:15:00Z", "predicted_load_kw": 160.0, "p10_kw": 150.0, "p90_kw": 170.0, "fingerprint_mean_kw": 158.0},
        ],
        confidence="medium",
        model_name="tree",
    )

    assert payload["horizon_steps"] == 4
    assert "aggregated_summary" in payload
    assert payload["summary"]["total_energy_kwh"] == 130.0
```

- [ ] **Step 2: Run the response-format test to verify it fails**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_response_formatter.py`

Expected: FAIL because the formatter does not exist.

- [ ] **Step 3: Implement the formatter and error helpers**

```python
def build_forecast_response(...):
    normalized_horizon, horizon_steps = normalize_horizon(horizon)
    aggregated_summary = build_aggregated_summary(slot_predictions)
    total_energy_kwh = round(sum(item["predicted_load_kw"] * 0.25 for item in slot_predictions), 3)
    return {
        "entity_type": entity_type,
        "entity_id": entity_id,
        "latest_input_timestamp": latest_timestamp,
        "lookback_days": 5,
        "lookback_steps": lookback_steps,
        "base_resolution": "15min",
        "horizon": normalized_horizon,
        "horizon_steps": horizon_steps,
        "model_name": model_name,
        "slot_predictions": slot_predictions,
        "aggregated_summary": aggregated_summary,
        "summary": {
            "mean_load_kw": ...,
            "peak_load_kw": ...,
            "peak_time": ...,
            "total_energy_kwh": total_energy_kwh,
            "confidence": confidence,
        },
    }
```

- [ ] **Step 4: Run the response-format tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_response_formatter.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/response_formatter.py \
  backend/tests/test_response_formatter.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add canonical forecast response formatter"
```

## Task 8: Replace the Forecasting API Surface

**Files:**
- Modify: `backend/app/api/forecasting.py`
- Create: `backend/app/api/explainability.py`
- Modify: `backend/app/api/router.py`
- Test: `backend/tests/test_forecasting.py`

- [ ] **Step 1: Write the failing API tests**

```python
@pytest.mark.anyio
async def test_forecasting_api_supports_target_aware_horizons(monkeypatch, tmp_path):
    history_csv = tmp_path / "ami_history.csv"
    fingerprint_csv = tmp_path / "fingerprints.csv"
    history = generate_ami_history(days=21, seed=7)
    history.to_csv(history_csv, index=False)
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(history_csv))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        await client.post("/api/v1/forecast/history/generate?days=21&seed=7")
        await client.post("/api/v1/forecast/fingerprint/build")
        await client.post("/api/v1/forecast/train?entity_type=feeder&entity_id=F_RES_01&horizon=1h&model=tree")
        response = await client.get("/api/v1/forecast/feeder?feeder_id=F_RES_01&horizon=1h&model=tree")

    assert response.status_code == 200
    payload = response.json()
    assert payload["forecast"]["horizon"] == "1h"
    assert len(payload["forecast"]["slot_predictions"]) == 4
```

- [ ] **Step 2: Run the API tests to verify they fail**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecasting.py`

Expected: FAIL because the old `/forecast/run?zone_id=...` API still exists.

- [ ] **Step 3: Implement the new forecasting and explanation routers**

```python
@router.post("/forecast/history/generate")
async def generate_forecast_history(days: int = Query(default=30, ge=7), seed: int = Query(default=7)) -> dict:
    history = generate_ami_history(days=days, seed=seed)
    save_ami_history(history)
    return {"status": "ok", "rows": len(history)}


@router.get("/forecast/feeder")
async def run_feeder_forecast(
    feeder_id: str = Query(..., min_length=1),
    horizon: str = Query(...),
    model: str = Query(default="tree"),
) -> dict:
    history = load_ami_history_or_raise()
    forecast = get_forecaster(model).predict(history, entity_type="feeder", entity_id=feeder_id, horizon=horizon)
    return {"status": "ok", "forecast": forecast}
```

- [ ] **Step 4: Run the API tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecasting.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/api/forecasting.py \
  backend/app/api/explainability.py \
  backend/app/api/router.py \
  backend/tests/test_forecasting.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add forecasting-only API surface"
```

## Task 9: Add Forecast Metrics and Peak-Error Evaluation

**Files:**
- Modify: `backend/app/forecasting/metrics.py`
- Modify: `backend/tests/test_tree_forecaster.py`

- [ ] **Step 1: Write the failing metrics test**

```python
from app.forecasting.metrics import calculate_regression_metrics


def test_metrics_include_peak_errors() -> None:
    metrics = calculate_regression_metrics(
        actual=[100.0, 120.0, 180.0, 140.0],
        predicted=[98.0, 118.0, 170.0, 145.0],
    )

    assert {"mae", "rmse", "mape", "r2", "peak_time_error", "peak_load_error"} <= set(metrics)
```

- [ ] **Step 2: Run the metrics test to verify it fails**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tree_forecaster.py -k metrics`

Expected: FAIL because the current metrics omit peak error fields.

- [ ] **Step 3: Implement the peak-error metrics**

```python
def calculate_regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    peak_time_error = float(abs(int(actual_array.argmax()) - int(predicted_array.argmax())))
    peak_load_error = float(abs(actual_array.max() - predicted_array.max()))
    return {
        "mae": ...,
        "rmse": ...,
        "mape": ...,
        "r2": ...,
        "peak_time_error": peak_time_error,
        "peak_load_error": peak_load_error,
    }
```

- [ ] **Step 4: Run the metrics tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tree_forecaster.py -k metrics`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/metrics.py \
  backend/tests/test_tree_forecaster.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add peak error forecast metrics"
```

## Task 10: Add Tree-Forecast Explanation with SHAP Fallback

**Files:**
- Create: `backend/app/explainability/shap_forecast_explainer.py`
- Create: `backend/tests/test_forecast_explainability.py`

- [ ] **Step 1: Write the failing explainability tests**

```python
from app.explainability.shap_forecast_explainer import explain_tree_forecast
from app.forecasting.tree_forecaster import TreeForecaster
from app.simulator.ami_history_generator import generate_ami_history


def test_explain_tree_forecast_falls_back_when_shap_is_unavailable(tmp_path, monkeypatch) -> None:
    history = generate_ami_history(days=21, seed=7)
    forecaster = TreeForecaster(model_dir=tmp_path)
    forecaster.train(history, entity_type="feeder", entity_id="F_RES_01", horizon="1h")
    monkeypatch.setitem(__import__("sys").modules, "shap", None)

    explanation = explain_tree_forecast(
        forecaster=forecaster,
        dataframe=history,
        entity_type="feeder",
        entity_id="F_RES_01",
        horizon="1h",
    )

    assert explanation["target"] == "peak"
    assert explanation["method"] in {"shap", "permutation_importance"}
    assert explanation["top_features"]
    assert explanation["plain_language_explanation"]
```

- [ ] **Step 2: Run the explainability tests to verify they fail**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecast_explainability.py`

Expected: FAIL because the explainer does not exist.

- [ ] **Step 3: Implement the explainer**

```python
def explain_tree_forecast(..., target: str = "peak") -> dict[str, Any]:
    artifact = forecaster.load_artifact(...)
    feature_frame = build_latest_inference_features(...)
    try:
        import shap
        explainer = shap.TreeExplainer(artifact["model"])
        values = explainer.shap_values(feature_frame)
        method = "shap"
    except Exception:
        values = permutation_feature_importance(artifact["model"], feature_frame)
        method = "permutation_importance"
    top_features = rank_feature_effects(values, MODEL_FEATURE_COLUMNS)
    return {
        "method": method,
        "target": target,
        "top_features": top_features,
        "plain_language_explanation": build_plain_language_explanation(...),
    }
```

- [ ] **Step 4: Run the explainability tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_forecast_explainability.py`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/explainability/shap_forecast_explainer.py \
  backend/tests/test_forecast_explainability.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add tree forecast explainability"
```

## Task 11: Update the Existing Static Frontend to the Forecasting-Only Flow

**Files:**
- Modify: `frontend/app.js`
- Modify: `frontend/index.html`
- Modify: `frontend/styles.css`
- Test: `backend/tests/test_dashboard_deployment.py`

- [ ] **Step 1: Write the failing dashboard deployment test**

```python
def test_frontend_serves_forecasting_labels() -> None:
    html = Path("/home/lets-smile/Documents/PulseGrid/gridpulse/frontend/index.html").read_text()
    assert "Substation Forecast" in html or "Forecast Explanation" in html
```

- [ ] **Step 2: Run the dashboard test to verify it fails**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_dashboard_deployment.py -k forecasting`

Expected: FAIL because the frontend still emphasizes the broader grid-ops dashboard.

- [ ] **Step 3: Implement the forecasting-only frontend flow**

```javascript
const PAGES = [
  "Substation Forecast",
  "Feeder Forecast",
  "Enterprise Forecast",
  "Fingerprint Heatmap",
  "Forecast Explanation",
];

async function loadForecast(entityType, entityId, horizon, model) {
  return apiGet(`/api/v1/forecast/${entityType}?${entityType}_id=${encodeURIComponent(entityId)}&horizon=${horizon}&model=${model}`);
}
```

- [ ] **Step 4: Run the dashboard tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_dashboard_deployment.py -k forecasting`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  frontend/index.html \
  frontend/app.js \
  frontend/styles.css \
  backend/tests/test_dashboard_deployment.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: add forecasting-only dashboard flow"
```

## Task 12: Integrate Optional TCN and N-HiTS with the New 15-Minute Interface

**Files:**
- Modify: `backend/app/forecasting/tcn.py`
- Create: `backend/app/forecasting/nhits.py`
- Modify: `backend/app/forecasting/model_registry.py`
- Modify: `backend/tests/test_tcn.py`

- [ ] **Step 1: Write the failing optional-model integration tests**

```python
def test_tcn_registry_uses_target_aware_horizon_strings(tmp_path):
    pytest.importorskip("torch")
    model = get_forecaster("tcn", model_dir=tmp_path)
    history = generate_ami_history(days=21, seed=7)
    train_result = model.train(history, entity_type="feeder", entity_id="F_RES_01", horizon="1h")
    assert train_result["artifact_path"].endswith("tcn_feeder_F_RES_01_1h.pt")
```

- [ ] **Step 2: Run the optional-model tests to verify they fail**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tcn.py`

Expected: FAIL because the existing TCN still assumes zone-hour inputs.

- [ ] **Step 3: Adapt TCN and add the `nhits.py` compatibility file**

```python
class TCNForecaster(BaseForecaster):
    def train(self, dataframe: pd.DataFrame, entity_type: str, entity_id: str, horizon: str) -> dict[str, Any]:
        normalized_horizon, horizon_steps = normalize_horizon(horizon)
        dataset = build_sequence_dataset(dataframe, entity_type=entity_type, entity_id=entity_id, lookback_steps=480, horizon_steps=horizon_steps)
        ...
```

- [ ] **Step 4: Run the optional-model tests again**

Run: `cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend && ./.venv/bin/python -m pytest -q tests/test_tcn.py`

Expected: PASS when `torch` is installed, or SKIP cleanly otherwise.

- [ ] **Step 5: Commit**

```bash
git -C /home/lets-smile/Documents/PulseGrid/gridpulse add \
  backend/app/forecasting/tcn.py \
  backend/app/forecasting/nhits.py \
  backend/app/forecasting/model_registry.py \
  backend/tests/test_tcn.py
git -C /home/lets-smile/Documents/PulseGrid/gridpulse commit -m "feat: adapt optional neural forecasters to 15-minute forecasting"
```

## Verification Sweep

- [ ] Run focused backend tests after each task.
- [ ] Run the full backend suite before completion:

```bash
cd /home/lets-smile/Documents/PulseGrid/gridpulse/backend
./.venv/bin/python -m pytest -q
```

- [ ] Run frontend syntax verification after dashboard changes:

```bash
node --check /home/lets-smile/Documents/PulseGrid/gridpulse/frontend/app.js
```

- [ ] If Docker validation is required, re-check compose config:

```bash
cd /home/lets-smile/Documents/PulseGrid/gridpulse
docker compose config
```

## Self-Review

Spec coverage:

- canonical 15-minute schema: covered in Tasks 1 and 2
- fingerprint DB: Task 3
- leakage-safe features and 480-step rolling inference: Task 4
- fingerprint baseline and degraded fallback: Task 5
- tree forecasting and structured slot outputs: Tasks 6 and 7
- forecasting APIs: Task 8
- metrics: Task 9
- SHAP fallback explanation: Task 10
- dashboard flow: Task 11
- optional TCN/N-HiTS integration: Task 12

Known repo-specific constraint:

- this repository currently lacks git author configuration, so commit steps will fail until `git config user.name` and `git config user.email` are set.
