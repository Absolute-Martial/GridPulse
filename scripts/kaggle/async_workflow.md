# GridPulse Async Kaggle Workflow

This is the intended non-blocking workflow for offloading training to Kaggle.

## 1. Build local forecasting inputs

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/history/generate?days=30&seed=7"
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/fingerprint/build"
```

## 2. Export the training bundle

```bash
./scripts/kaggle/export_training_bundle.sh
```

This exports:

- `kaggle/dataset/ami_history.csv`
- `kaggle/dataset/feeder.csv`
- `kaggle/dataset/substation.csv`
- `kaggle/dataset/fingerprints.csv`

Use `kaggle/dataset/run_config.json` to decide which file and target the Kaggle
trainer should run.

## 3. Version the Kaggle dataset

```bash
kaggle datasets version -p ./kaggle/dataset -m "updated forecasting inputs"
```

## 4. Push the training kernel

```bash
kaggle kernels push -p ./kaggle/kernel --timeout 3600
```

If you want a specific job from `run_config.json`, set:

```bash
export GRIDPULSE_JOB_NAME=feeder_fd_res_01_1h
```

## 5. Poll job status

```bash
kaggle kernels status your-kaggle-username/gridpulse-forecast-train
```

## 6. Download outputs

```bash
kaggle kernels output your-kaggle-username/gridpulse-forecast-train -p ./kaggle/output -o
```

## 7. Import artifacts locally

```bash
./scripts/kaggle/import_artifacts.sh
```

## Runtime rule

The backend keeps serving local forecasts from the most recent local artifact.
It must not wait for Kaggle to finish.
