# Secondary Substation Forecasting MVP Design

## Goal

Refocus GridPulse on an offline forecasting MVP that uses canonical 15-minute AMI/SMU history to forecast downstream load for substations, feeders, customer groups, and enterprise dedicated lines, with explanation support for tree-based models.

## Scope

Active scope for this phase:

- synthetic AMI history generation
- feeder fingerprint database
- forecasting feature engineering
- forecasting model registry
- fingerprint baseline model
- tree-based forecasting model
- TCN integration if already scaffolded
- N-HiTS integration if already scaffolded
- forecast evaluation metrics
- SHAP or permutation-importance explanations for tree models
- forecasting and explainability API endpoints

Inactive scope for this phase:

- anomaly detection
- alerts
- GNN risk prediction
- pandapower OPF
- routing
- dynamic allocation
- secondary allocation
- recommendation engine
- Monte Carlo risk or allocation endpoints
- control decisions

Inactive modules may remain in the repository, but the forecasting flow must not call them.

## Architecture Decision

Use one canonical 15-minute forecasting stack.

- Do not create separate hourly datasets.
- Do not create separate architectures for substation, feeder, and enterprise forecasting.
- Treat entity type as a forecast target dimension handled by one forecasting engine.
- Use the same response contract across target types.

## Canonical Data Source

The local AMI/SMU history CSV is the source of truth.

Base resolution:

- `15min`

The forecasting system trains and predicts from 15-minute history only. Any hourly summaries are derived helpers and must never replace the slot-level output.

## Canonical Schema

Each record must support the following fields:

- `timestamp`
- `entity_type`
- `entity_id`
- `secondary_substation_id`
- `transformer_id`
- `feeder_id`
- `feeder_type`
- `customer_group_id`
- `customer_type`
- `enterprise_id`
- `is_dedicated_line`
- `contracted_md_kw`
- `load_kw`
- `interval_energy_kwh`
- `temperature_c`
- `humidity_percent`
- `day_type`
- `hour`
- `minute`
- `slot_index`
- `month`
- `day_of_week`
- `season`
- `season_index`
- `is_weekend`
- `is_holiday`
- `data_quality_flag`
- `source_type`
- `production_schedule_kw`
- `fingerprint_mean_kw`
- `fingerprint_p10_kw`
- `fingerprint_p90_kw`

If the source contains `interval_energy_kwh` instead of `load_kw`, convert it as:

`load_kw = interval_energy_kwh / 0.25`

because one 15-minute interval is `0.25` hour.

15-minute AMI is the only canonical resolution in this phase.

## Season Mapping

Season is additive metadata and must not replace `month`.

Season mapping for Nepal-style synthetic data:

- winter: December, January, February
- spring: March, April
- summer: May, June
- monsoon: July, August, September
- autumn: October, November

Season index mapping:

- `0 = winter`
- `1 = spring`
- `2 = summer`
- `3 = monsoon`
- `4 = autumn`

`month` remains in the schema to preserve finer annual cycle behavior, while `season` and `season_index` capture broader weather and demand patterns.

## Forecast Targets

The same engine must forecast for:

- secondary substation level
- feeder level
- customer group level
- enterprise dedicated line level

The selected target is defined by API inputs and passed internally as metadata and filter criteria. Forecast outputs must remain structurally consistent across target types.

## Forecast Horizons

Accepted public API horizon values:

- `1h`
- `4h`
- `24h`

Normalized internal mapping:

- `1h -> 4` future 15-minute slots
- `4h -> 16` future 15-minute slots
- `24h -> 96` future 15-minute slots

Raw integer horizon inputs must not be the public contract unless normalized to the canonical string format first.

## Rolling Inference Contract

Each rolling forecast cycle must:

1. fetch the latest timestamp from the local IoT adapter or AMI history source
2. load the latest 5 days of 15-minute history for the selected target
3. build a feature window over those 480 time steps
4. predict the future slot sequence for the requested horizon
5. return slot-level forecasts and optional aggregated summaries

Lookback rules:

- `lookback_days = 5`
- `lookback_steps = 480`
- forecast anchor = latest available timestamp in local history

Degraded inference rules:

- if `480` or more historical steps are available, run the standard rolling forecast
- if fewer than `480` but at least `96` steps are available, run a degraded forecast and mark confidence as `low`
- if fewer than `96` steps are available, fall back to `fingerprint_baseline` if available
- if fewer than `96` steps are available and no fingerprint profile exists, return a structured insufficient-history or missing-fingerprint error

For a latest timestamp of `2026-05-25T10:15:00Z`, a `1h` forecast must predict:

- `2026-05-25T10:30:00Z`
- `2026-05-25T10:45:00Z`
- `2026-05-25T11:00:00Z`
- `2026-05-25T11:15:00Z`

## Forecast Output Contract

Mandatory output:

- `slot_predictions`

Optional derived helper:

- `aggregated_summary`

Slot predictions are mandatory for all horizons:

- `1h`: 4 slots
- `4h`: 16 slots
- `24h`: 96 slots

Hourly summaries are convenience aggregates only:

- `1h`: optional 1 summary row
- `4h`: optional 4 summary rows
- `24h`: optional 24 summary rows

Representative response shape:

```json
{
  "entity_type": "feeder",
  "feeder_id": "F_RES_01",
  "latest_input_timestamp": "2026-05-25T10:15:00Z",
  "lookback_days": 5,
  "lookback_steps": 480,
  "base_resolution": "15min",
  "horizon": "1h",
  "horizon_steps": 4,
  "model_name": "tree",
  "slot_predictions": [
    {
      "timestamp": "2026-05-25T10:30:00Z",
      "predicted_load_kw": 142.5,
      "p10_kw": 130.2,
      "p90_kw": 156.8,
      "fingerprint_mean_kw": 139.7
    }
  ],
  "aggregated_summary": [
    {
      "hour_start": "2026-05-25T10:30:00Z",
      "mean_load_kw": 149.35,
      "peak_load_kw": 155.2,
      "total_energy_kwh": 149.35
    }
  ],
  "summary": {
    "mean_load_kw": 149.35,
    "peak_load_kw": 155.2,
    "peak_time": "2026-05-25T11:15:00Z",
    "total_energy_kwh": 149.35,
    "confidence": "medium"
  }
}
```

`total_energy_kwh` must be computed as:

`sum(predicted_load_kw * 0.25)`

## Forecasting Models

The system must expose one forecasting interface with a pluggable registry.

Models in scope:

- `fingerprint_baseline`
- `tree`
- `tcn` if available
- `nhits` if available

Model selection may vary by:

- `model_name`
- normalized horizon
- entity type

The output contract must not vary by model choice.

## Fingerprint Baseline

The fingerprint database stores historical baseline behavior at 15-minute resolution. It must support at least:

- `fingerprint_mean_kw`
- `fingerprint_p10_kw`
- `fingerprint_p90_kw`

The baseline model uses those profiles directly for deterministic forecast and confidence reference values.

## Feature Engineering

Feature engineering must work at 15-minute resolution and support rolling inference from the latest 480 steps.

Leakage rule:

- no future measured `load_kw` may be used as an input feature at train time or inference time
- all lag, rolling, fingerprint, and contextual features must be built only from information available at or before the forecast anchor timestamp

Feature groups:

- target identity metadata
- calendar features
- seasonal features
- historical lag features
- rolling statistics
- weather features
- fingerprint baseline features
- production schedule features

Examples:

- lagged load values
- rolling means over recent slot windows
- hour, month, season, season index
- weekend and holiday indicators
- contracted maximum demand
- forecast-time fingerprint mean and quantiles

## Training and Evaluation

Training must remain fully offline.

Evaluation metrics:

- `MAE`
- `RMSE`
- `MAPE`
- `R2`
- `peak_time_error`
- `peak_load_error`

The response formatter and API layer must also return consistent structured error payloads for:

- invalid horizon
- missing entity
- insufficient history
- missing fingerprint
- untrained model

Training and inference must support the same target types through one engine.

## Explanation Layer

Explanation scope is limited to forecast explanation for tree-based models.

Primary explanation method:

- SHAP

Fallback if SHAP is unavailable:

- permutation importance

Do not force SHAP onto TCN or N-HiTS in this phase.

Default explanation target for tree-based forecasts:

- explain the `peak` prediction by default

Explanation responses must return:

- top contributing features
- feature importance magnitudes
- plain-language explanation text tied to the selected target and forecast window

Representative explanation style:

`Feeder F_RES_01 is forecasted to peak at 19:00 because historical evening demand is high, temperature is above normal, and the same time slot last week showed similar growth.`

## API Endpoints

Forecasting endpoints:

- `POST /api/v1/forecast/history/generate`
- `POST /api/v1/forecast/fingerprint/build`
- `POST /api/v1/forecast/train`
- `GET /api/v1/forecast/substation?substation_id=SS_01&horizon=24h`
- `GET /api/v1/forecast/feeder?feeder_id=F_RES_01&horizon=24h`
- `GET /api/v1/forecast/enterprise?enterprise_id=ENT_01&horizon=24h`
- `GET /api/v1/forecast/evaluate?...`

Explainability endpoint:

- `GET /api/v1/explain/forecast?...`

API rules:

- horizon input must accept `1h`, `4h`, and `24h`
- responses must return slot-level predictions for every horizon
- aggregated summaries may be returned as a derived helper only
- inactive modules must not be called from any forecasting or explanation endpoint

Structured API errors must distinguish:

- invalid horizon
- missing entity
- insufficient history
- missing fingerprint
- untrained model

## File Responsibilities

Planned active files:

- `backend/app/simulator/ami_history_generator.py`
- `backend/app/forecasting/feeder_fingerprint.py`
- `backend/app/forecasting/features.py`
- `backend/app/forecasting/base.py`
- `backend/app/forecasting/fingerprint_baseline.py`
- `backend/app/forecasting/tree_forecaster.py`
- `backend/app/forecasting/tcn.py`
- `backend/app/forecasting/nhits.py`
- `backend/app/forecasting/model_registry.py`
- `backend/app/forecasting/metrics.py`
- `backend/app/explainability/shap_forecast_explainer.py`
- `backend/app/api/forecasting.py`
- `backend/app/api/explainability.py`

Existing inactive modules may remain present but are not part of the active implementation surface for this phase.

## Implementation Order

Implement the active forecasting MVP in this order:

1. schema validator
2. synthetic AMI generator
3. fingerprint builder
4. feature builder
5. fingerprint baseline
6. tree forecaster
7. response formatter
8. APIs
9. metrics
10. explanation
11. dashboard
12. optional TCN and N-HiTS integration

## Testing Requirements

Add tests for:

- synthetic AMI history generation
- fingerprint database build
- feature engineering
- model train and predict behavior
- forecast output format
- SHAP explanation fallback behavior
- forecast API endpoints

## Runtime Constraints

- offline only after installation
- no cloud APIs
- no live IoT hardware requirement
- no automatic control actions
- no automatic alerting
- no invocation of inactive decision modules
