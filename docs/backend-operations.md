# GridPulse Backend Operations Guide

## Purpose

This guide explains how to operate the current GridPulse backend directly
through its HTTP API.

The active backend scope is the forecasting MVP:

- generate canonical 15-minute AMI history
- build fingerprint profiles
- train forecasting models
- run forecasts for substations, feeders, and enterprise lines
- evaluate forecast quality
- explain tree-based forecasts

## Start the backend

Local development:

```bash
cd gridpulse/backend
.venv/bin/python -m uvicorn app.main:app --reload
```

Docker:

```bash
cd gridpulse
docker compose up --build
```

Backend base URL:

`http://127.0.0.1:8000`

## Active backend scope

Use this guide for the current forecasting MVP only.

Active:

- canonical 15-minute AMI history
- fingerprint build and lookup
- forecasting model training
- substation, feeder, and enterprise forecasting
- forecast evaluation
- tree forecast explanation

Inactive in this phase:

- anomaly and alerting
- GNN and topology risk
- OPF and pandapower validation
- routing, recommendation, allocation, and control

Those older modules may still exist in the repo, but they are not part of the
active forecasting operating flow.

## Streamlit Cloud demo

GridPulse also includes a self-contained Streamlit demo for the current
hackathon setup. It loads the packaged `.pkl` artifact from
`data/models/forecasting/`, simulates a live 15-minute AMI stream, and renders
the forecast with p10/p90 bands.

Run it locally with:

```bash
cd gridpulse
streamlit run frontend/streamlit_app.py
```

The Streamlit demo does not call the backend API. It is intended for Streamlit
Cloud deployment with the repository-level `requirements.txt`.

Research reference:

- [research-reference.md](./research-reference.md)

## Recommended operator flow

Run the backend in this order:

1. generate AMI history
2. build fingerprints
3. train the model you want to use
4. run forecasts
5. run evaluation
6. run explanation for tree forecasts

## Health check

`GET /health`

```bash
curl http://127.0.0.1:8000/health
```

Use this to confirm the FastAPI process is alive before running forecasting
jobs.

## Forecasting route map

Core routes:

- `POST /api/v1/forecast/history/generate`
- `POST /api/v1/forecast/fingerprint/build`
- `GET /api/v1/forecast/fingerprint`
- `POST /api/v1/forecast/train`
- `GET /api/v1/forecast/substation`
- `GET /api/v1/forecast/feeder`
- `GET /api/v1/forecast/enterprise`
- `GET /api/v1/forecast/evaluate`
- `GET /api/v1/forecast/models`
- `GET /api/v1/explain/forecast`

Canonical horizon input:

- `1h`
- `4h`
- `24h`

Internal slot mapping:

- `1h = 4` slots
- `4h = 16` slots
- `24h = 96` slots

## Generate AMI history

`POST /api/v1/forecast/history/generate?days=30&seed=7`

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/forecast/history/generate?days=30&seed=7"
```

What it does:

- generates canonical 15-minute AMI/SMU history
- writes it to the local AMI history file

Key response fields:

- `status`
- `rows`
- `path`

## Build fingerprint database

`POST /api/v1/forecast/fingerprint/build`

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/forecast/fingerprint/build"
```

What it does:

- reads the AMI history
- groups by `entity_type`, `entity_id`, `day_of_week`, `slot_index`
- computes `fingerprint_mean_kw`, `fingerprint_p10_kw`, `fingerprint_p90_kw`

This step is required for:

- `fingerprint_baseline` forecasts
- degraded fallback when history is too short
- side-by-side baseline comparison in forecast responses

## Inspect fingerprint rows

`GET /api/v1/forecast/fingerprint?entity_type=feeder&entity_id=FD_RES_01`

```bash
curl \
  "http://127.0.0.1:8000/api/v1/forecast/fingerprint?entity_type=feeder&entity_id=FD_RES_01"
```

Use this to inspect whether the baseline profile exists for the target you want
to forecast.

## Train a forecasting model

Supported public models:

- `tree`
- `fingerprint_baseline`

Current recommendation:

- use `tree` for the main trained forecast path
- keep `fingerprint_baseline` as the cold-start or degraded fallback path

Example:

`POST /api/v1/forecast/train?feeder_id=FD_RES_01&horizon=1h&model=tree`

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/forecast/train?feeder_id=FD_RES_01&horizon=1h&model=tree"
```

Valid horizons:

- `1h`
- `4h`
- `24h`

Valid target selectors:

- `substation_id`
- `feeder_id`
- `enterprise_id`

Generic target form is also supported:

- `entity_type`
- `entity_id`

Example generic form:

```bash
curl -X POST \
  "http://127.0.0.1:8000/api/v1/forecast/train?entity_type=feeder&entity_id=FD_RES_01&horizon=4h&model=tree"
```

## Run a forecast

Examples:

Substation:

```bash
curl \
  "http://127.0.0.1:8000/api/v1/forecast/substation?substation_id=SS_KTM_01&horizon=24h&model=tree"
```

Feeder:

```bash
curl \
  "http://127.0.0.1:8000/api/v1/forecast/feeder?feeder_id=FD_RES_01&horizon=4h&model=tree"
```

Enterprise:

```bash
curl \
  "http://127.0.0.1:8000/api/v1/forecast/enterprise?enterprise_id=ENT_CEMENT_01&horizon=1h&model=tree"
```

Response shape:

- `forecast.entity_type`
- `forecast.entity_id`
- `forecast.latest_input_timestamp`
- `forecast.lookback_days`
- `forecast.lookback_steps`
- `forecast.base_resolution`
- `forecast.horizon`
- `forecast.horizon_steps`
- `forecast.model_name`
- `forecast.slot_predictions`
- `forecast.aggregated_summary`
- `forecast.summary`

Important:

- `slot_predictions` is the main output
- `aggregated_summary` is derived reporting output

Shortened response example:

```json
{
  "status": "ok",
  "forecast": {
    "entity_type": "feeder",
    "entity_id": "FD_RES_01",
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
        "timestamp": "2026-05-25T10:00:00Z",
        "mean_predicted_load_kw": 149.35,
        "total_energy_kwh": 37.34
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
}
```

Confidence behavior:

- `>= 480` rows for the target: standard rolling forecast path
- `96-479` rows: degraded forecast with low confidence
- `< 96` rows: fallback to `fingerprint_baseline` if fingerprint data exists

## Evaluate a model

`GET /api/v1/forecast/evaluate?feeder_id=FD_RES_01&horizon=1h&model=tree`

```bash
curl \
  "http://127.0.0.1:8000/api/v1/forecast/evaluate?feeder_id=FD_RES_01&horizon=1h&model=tree"
```

Current metrics:

- `mae`
- `rmse`
- `mape`
- `r2`
- `peak_time_error`
- `peak_load_error`

Use evaluation after training to compare:

- one entity across multiple horizons
- tree model versus fingerprint baseline
- feeder versus enterprise behavior

## List the active models

`GET /api/v1/forecast/models`

```bash
curl "http://127.0.0.1:8000/api/v1/forecast/models"
```

This returns the models currently exposed by the active forecasting API.

## Explain a tree forecast

`GET /api/v1/explain/forecast?feeder_id=FD_RES_01&horizon=1h&model=tree`

```bash
curl \
  "http://127.0.0.1:8000/api/v1/explain/forecast?feeder_id=FD_RES_01&horizon=1h&model=tree"
```

What it returns:

- top contributing features
- importance values
- plain-language explanation
- forecast peak summary context

This endpoint is intentionally limited to `model=tree` in the current phase.

## Storage locations

Environment overrides:

```bash
export GRIDPULSE_AMI_HISTORY_PATH=/absolute/path/to/ami_history.csv
export GRIDPULSE_FORECAST_FINGERPRINT_PATH=/absolute/path/to/fingerprints.csv
export GRIDPULSE_FORECAST_ARTIFACT_DIR=/absolute/path/to/forecasting-artifacts
```

Default purpose:

- `GRIDPULSE_AMI_HISTORY_PATH`
  canonical AMI history store
- `GRIDPULSE_FORECAST_FINGERPRINT_PATH`
  fingerprint database store
- `GRIDPULSE_FORECAST_ARTIFACT_DIR`
  trained forecasting artifact directory

Default repo-local paths:

- `data/forecasting/ami_history.csv`
- `data/forecasting/fingerprints.csv`
- `data/models/forecasting/`

## Structured errors

The backend returns structured forecasting errors with:

- `detail.error`
- `detail.message`

Common error codes:

- `invalid_horizon`
- `missing_entity`
- `missing_history`
- `insufficient_history`
- `missing_fingerprint`
- `untrained_model`
- `invalid_model`

Typical meanings:

- `missing_history`
  AMI history has not been generated or loaded yet
- `insufficient_history`
  the selected target does not have enough rows for the requested forecast path
- `missing_fingerprint`
  fingerprint data is required but has not been built yet
- `untrained_model`
  the selected trained artifact does not exist yet

Example error payload:

```json
{
  "detail": {
    "error": "missing_fingerprint",
    "message": "No fingerprint database is available. Build it first."
  }
}
```

## Operational recommendation

For normal usage, treat the backend as a control plane around three local state
files:

- AMI history CSV
- fingerprint CSV
- trained artifact files

If one of those is missing, rebuild it in that order instead of trying to
debug the later steps first.

## Minimal end-to-end operator sequence

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/history/generate?days=30&seed=7"
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/fingerprint/build"
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/train?feeder_id=FD_RES_01&horizon=1h&model=tree"
curl "http://127.0.0.1:8000/api/v1/forecast/feeder?feeder_id=FD_RES_01&horizon=1h&model=tree"
curl "http://127.0.0.1:8000/api/v1/forecast/evaluate?feeder_id=FD_RES_01&horizon=1h&model=tree"
curl "http://127.0.0.1:8000/api/v1/explain/forecast?feeder_id=FD_RES_01&horizon=1h&model=tree"
```
