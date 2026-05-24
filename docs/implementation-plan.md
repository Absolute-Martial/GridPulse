# GridPulse Current Implementation Plan

## Implemented

- canonical 15-minute schema validation
- deterministic synthetic AMI history generation
- fingerprint database builder
- target-aware leakage-safe feature builder
- fingerprint baseline forecasting model
- tree-based forecasting model
- slot-level forecast response formatter
- forecasting API endpoints
- tree forecast explanation with SHAP fallback
- forecasting-only static dashboard

## Current limitations

- public target-aware API currently exposes `tree` and `fingerprint_baseline`
- optional neural models remain in the repository but are not the main active
  API path
- the repository still contains older grid, anomaly, optimization, and
  recommendation modules that are intentionally inactive in this phase

## Next implementation priorities

1. finish target-aware public API exposure for `TCN` and `N-HiTS`
2. add artifact versioning for trained forecasting models
3. add export/import helpers for Kaggle-trained artifacts
4. extend evaluation reports with per-target comparison tables
