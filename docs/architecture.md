# GridPulse Architecture

## Current system

GridPulse is currently implemented as an offline forecasting MVP centered on
canonical 15-minute AMI/SMU history. The active system scope is forecasting and
forecast explanation. Older smart-grid modules remain in the repository but are
not part of the active execution path.

## Active backend architecture

- `backend/app/simulator/ami_history_generator.py`
  Generates deterministic synthetic 15-minute AMI history for substations,
  feeders, customer groups, and enterprise dedicated lines.
- `backend/app/forecasting/schema.py`
  Validates the canonical forecasting schema, enforces 15-minute resolution, and
  normalizes horizons.
- `backend/app/forecasting/feeder_fingerprint.py`
  Builds fingerprint statistics by `entity_type`, `entity_id`, `day_of_week`,
  and `slot_index`.
- `backend/app/forecasting/features.py`
  Produces leakage-safe target-aware lag and rolling features from canonical AMI
  history.
- `backend/app/forecasting/fingerprint_baseline.py`
  Baseline forecaster using stored fingerprint profiles.
- `backend/app/forecasting/tree_forecaster.py`
  Tree-based forecaster for slot-level multi-step prediction.
- `backend/app/explainability/shap_forecast_explainer.py`
  Explains tree forecasts with SHAP when available and permutation importance
  fallback otherwise.
- `backend/app/api/forecasting.py`
  Forecast generation, training, evaluation, and fingerprint endpoints.
- `backend/app/api/explainability.py`
  Forecast explanation endpoint.

## Data flow

1. Generate or load canonical AMI history from local CSV.
2. Validate schema and infer `load_kw` from `interval_energy_kwh` when needed.
3. Build fingerprint database from history.
4. Build target-aware 15-minute features for one forecast entity.
5. Train or run the selected forecaster.
6. Return slot-level forecast output plus derived `aggregated_summary`.
7. Optionally explain the peak-focused tree forecast.

## Forecasting design

- Canonical history resolution: `15min`
- Horizons:
  - `1h = 4` future slots
  - `4h = 16` future slots
  - `24h = 96` future slots
- Rolling inference lookback:
  - preferred: `480` steps (`5` days)
  - degraded: `96-479` steps
  - fallback to fingerprint baseline: `<96` steps when fingerprint exists

## Inactive modules

These modules remain present for future phases but are inactive in the current
forecasting flow:

- anomaly detection
- alerts
- pandapower OPF
- GNN risk prediction
- routing
- allocation
- recommendation engine
- control actions
