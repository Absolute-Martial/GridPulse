# GridPulse Agent Rules

## Current system goal

GridPulse currently operates as an offline forecasting MVP for downstream
secondary-substation load prediction using canonical 15-minute AMI/SMU history.

## Active scope

- synthetic 15-minute AMI history generation
- fingerprint database build
- target-aware feature engineering
- fingerprint baseline forecasting
- tree-based forecasting
- forecast evaluation metrics
- tree forecast explanation with SHAP fallback
- forecasting-only API and dashboard

## Inactive scope

Do not activate or extend these modules in the current forecasting flow:

- anomaly detection
- alerts
- pandapower OPF
- GNN risk prediction
- routing
- allocation
- recommendation engine
- control actions

## Technical rules

- keep the system offline-first after installation
- do not use cloud APIs at runtime
- do not require live IoT hardware
- keep the canonical forecast resolution at `15min`
- keep horizons normalized as `1h`, `4h`, `24h`
- prevent feature leakage from future measured load values
- favor small, testable modules and explicit interfaces
- do not introduce Transformer-based forecasting models

## Current public forecasting models

- `fingerprint_baseline`
- `tree`

Optional neural modules may remain in the repository but are not the primary
public API path in the current phase.
