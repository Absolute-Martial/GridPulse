# GridPulse API Specification

## Active endpoints

### Forecast history

`POST /api/v1/forecast/history/generate?days=30&seed=7`

Generates canonical 15-minute AMI history and stores it locally.

### Fingerprint database

`POST /api/v1/forecast/fingerprint/build`

Builds the local fingerprint database from AMI history.

`GET /api/v1/forecast/fingerprint?entity_type=feeder&entity_id=FD_RES_01`

Returns stored fingerprint rows for one target.

### Forecasting

`POST /api/v1/forecast/train?feeder_id=FD_RES_01&horizon=1h&model=tree`

Trains the selected model for one target.

`GET /api/v1/forecast/substation?substation_id=SS_KTM_01&horizon=24h&model=tree`

`GET /api/v1/forecast/feeder?feeder_id=FD_RES_01&horizon=4h&model=tree`

`GET /api/v1/forecast/enterprise?enterprise_id=ENT_CEMENT_01&horizon=1h&model=tree`

Returns slot-level forecast output.

`GET /api/v1/forecast/evaluate?feeder_id=FD_RES_01&horizon=1h&model=tree`

Returns evaluation metrics for the selected target and horizon.

`GET /api/v1/forecast/models`

Returns the active public forecasting models:

- `fingerprint_baseline`
- `tree`

### Explainability

`GET /api/v1/explain/forecast?feeder_id=FD_RES_01&horizon=1h&model=tree`

Returns peak-target explanation for the current tree forecast.

## Forecast response contract

All forecast endpoints return:

- `entity_type`
- `entity_id`
- `latest_input_timestamp`
- `lookback_days`
- `lookback_steps`
- `base_resolution`
- `horizon`
- `horizon_steps`
- `model_name`
- `slot_predictions`
- `aggregated_summary`
- `summary`

`slot_predictions` is mandatory. `aggregated_summary` is derived and intended
for reporting or dashboard use.

## Structured errors

The forecasting surface returns structured error codes for:

- `invalid_horizon`
- `missing_entity`
- `missing_history`
- `insufficient_history`
- `missing_fingerprint`
- `untrained_model`
- `invalid_model`
