# Kaggle Training Guide

## Purpose

This document explains how to use Kaggle to train the current GridPulse
forecasting system without changing the live backend execution path.

The current active forecasting system is:

- canonical 15-minute AMI history
- fingerprint baseline forecaster
- tree-based forecaster
- tree forecast explanation

## Recommended Kaggle scope

Use Kaggle for:

- offline model experimentation
- tree model retraining on larger synthetic or curated AMI datasets
- feature comparison and error analysis
- optional neural model prototyping

Do not use Kaggle as the production runtime. The deployed GridPulse station
must remain offline after installation.

## Data to upload to Kaggle

Export the canonical AMI history CSV from:

`data/forecasting/ami_history.csv`

Optional:

- `data/forecasting/fingerprints.csv`
- trained artifacts from `data/models/forecasting/`

## Kaggle notebook workflow

1. Upload `ami_history.csv` to a Kaggle dataset.
2. In the notebook, install or import:
   - `pandas`
   - `numpy`
   - `scikit-learn`
   - `shap` if needed
3. Recreate the current preprocessing contract:
   - 15-minute rows only
   - same season mapping
   - same target identity fields
   - same lag and rolling features
4. Train the model:
   - baseline comparison against fingerprint mean
   - tree model as the primary production-compatible model
5. Evaluate:
   - `MAE`
   - `RMSE`
   - `MAPE`
   - `R2`
   - `peak_time_error`
   - `peak_load_error`
6. Export the trained artifact as `.pkl`
7. Download the artifact and copy it back to:
   - `data/models/forecasting/`
   - or `station-data/models/offline/` for offline deployment

## Minimal Kaggle training outline

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

history = pd.read_csv("/kaggle/input/gridpulse/ami_history.csv")

# Rebuild the same feature contract used by GridPulse before training.
# Train a direct multi-step regressor or one-step recursive regressor,
# then export the fitted artifact with pickle.
```

## Artifact handoff back to GridPulse

After Kaggle training:

1. Download the model artifact.
2. Place it in the local artifact directory used by the backend:
   - `data/models/forecasting/`
3. For offline station deployment, copy it into:
   - `station-data/models/offline/`
4. Restart the backend or containers if the runtime is already running.

## Practical recommendation

For the current system, Kaggle should be used to improve the `tree` model first.
That matches the production API already exposed by GridPulse today and avoids a
mismatch between research training and deployed inference.
