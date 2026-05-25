# Kaggle Compute Strategy for GridPulse

## Short answer

Use Kaggle for offline training and batch preprocessing, not as a live
always-on backend proxy.

That recommendation follows from the current Kaggle CLI model: Kaggle kernels
are pushed, run, monitored, and their outputs are later downloaded, which makes
them suitable for batch jobs rather than stable low-latency backend serving.
This is an inference from the official Kaggle CLI flow, not a direct Kaggle
policy statement.

For the current GridPulse forecasting MVP, the right model is:

- local backend API for forecasting operations
- Kaggle only for heavy training, comparison runs, and artifact generation

## Why Kaggle fits this project

GridPulse currently needs extra compute mainly for:

- heavier tree-model retraining on larger AMI history
- optional TCN or N-HiTS experimentation
- bulk feature generation and evaluation sweeps
- SHAP analysis on larger training slices

Those are batch workloads. Kaggle is well-aligned with batch workloads because
the official workflow is:

- upload dataset
- push kernel
- run kernel
- inspect status
- download output

## Why Kaggle is a weak live proxy

A live proxy would mean your backend forwards runtime inference calls to Kaggle
and waits for an immediate response. That is a poor fit for the current Kaggle
execution model because:

- the official kernel flow is push-and-run, not long-lived service hosting
- kernels are run with explicit timeout settings
- results are retrieved as output artifacts after execution
- the public CLI/docs describe job execution and output retrieval, not inbound
  HTTP service exposure

That means Kaggle is suitable for:

- training runs
- parameter sweeps
- feature experiments
- batch SHAP runs

It is not a good fit for:

- synchronous forecast API calls
- low-latency live inference
- always-on service proxying

Practical conclusion:

- do not build the current forecasting runtime around synchronous Kaggle calls
- use Kaggle to produce artifacts that your offline backend can consume later

## Recommended architecture

Use Kaggle as a remote batch worker and keep GridPulse as the local control
plane.

### Local GridPulse responsibilities

- generate and validate canonical AMI history
- build fingerprint database
- package training input
- trigger or prepare Kaggle jobs
- import trained artifacts back into the backend
- run local inference from downloaded artifacts

### Kaggle responsibilities

- train the heavy model
- run large evaluation jobs
- optionally compute explanation summaries
- export `.pkl`, metrics JSON, CSV reports, or plots

## Recommended operating model

Split the system into two planes.

### 1. Runtime plane

This stays local and offline:

- FastAPI backend
- local AMI history CSV
- local fingerprint CSV
- local trained artifacts
- local forecast API calls

### 2. Batch compute plane

This is optional and connected:

- Kaggle dataset packaging
- Kaggle notebook or script execution
- batch training output download
- local artifact import

This separation preserves the offline station requirement.

## Best practical pattern

### Pattern 1: Manual batch offload

This is the safest current approach.

1. Generate local AMI history:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/history/generate?days=30&seed=7"
```

2. Export or locate:

- `data/forecasting/ami_history.csv`
- optionally `data/forecasting/fingerprints.csv`

3. Upload the files to a Kaggle dataset.

4. Run a Kaggle notebook/script that:

- rebuilds GridPulse features
- trains the target model
- evaluates it
- writes outputs to `/kaggle/working/`

5. Download the artifacts.

6. Place them into:

- `data/models/forecasting/`
- or `station-data/models/offline/`

7. Run local inference through GridPulse.

### Pattern 2: Semi-automated CLI batch offload

Use the Kaggle CLI from a connected machine.

The official Kaggle CLI supports datasets and kernels, and the official kernel
docs show:

- `kaggle kernels init`
- `kaggle kernels push`
- `kaggle kernels status`
- `kaggle kernels output`

That means you can script a batch pipeline like:

1. write `ami_history.csv`
2. version a Kaggle dataset
3. push a training kernel
4. poll kernel status
5. download output artifacts
6. copy artifacts back into GridPulse

### Pattern 3: Async proxy helper

If you want a "proxy", make it asynchronous.

Good shape:

1. local GridPulse exports a training package
2. a helper process uploads it to Kaggle
3. Kaggle runs the training notebook
4. the helper polls job state
5. outputs are downloaded when complete
6. the helper copies artifacts into the local model directory
7. GridPulse keeps serving local forecasts from the last good artifact

Do not make the forecast endpoint block on Kaggle completion.

## Example Kaggle CLI workflow

### Create or version the dataset

```bash
kaggle datasets init -p ./kaggle/dataset
```

Add your exported files and metadata, then:

```bash
kaggle datasets create -p ./kaggle/dataset
```

For later updates:

```bash
kaggle datasets version -p ./kaggle/dataset -m "updated ami history"
```

### Initialize the training kernel

```bash
kaggle kernels init -p ./kaggle/kernel
```

The official kernel metadata supports fields such as:

- `code_file`
- `kernel_type`
- `enable_gpu`
- `enable_internet`
- `dataset_sources`

Then run:

```bash
kaggle kernels push -p ./kaggle/kernel --timeout 3600
```

Poll:

```bash
kaggle kernels status your-username/gridpulse-forecast-train
```

Download results:

```bash
kaggle kernels output your-username/gridpulse-forecast-train -p ./kaggle/output -o
```

Minimum practical local folder layout:

```text
kaggle/
├── dataset/
│   ├── ami_history.csv
│   ├── fingerprints.csv
│   ├── sample_ami_history.csv
│   ├── sample_fingerprints.csv
│   └── dataset-metadata.json
├── kernel/
│   ├── train_forecast.py
│   └── kernel-metadata.json
└── output/
```

## Parallel processing strategy

If your local hardware is weak, parallelize at the job level rather than trying
to create one giant notebook.

Recommended splits:

- by `entity_type`
- by horizon: `1h`, `4h`, `24h`
- by model family
- by fold or evaluation slice

Recommended first split for this repo:

- one job per horizon
- optionally one job per target family

Example:

- job A: substation `1h`
- job B: feeder `1h`
- job C: feeder `4h`
- job D: feeder `24h`
- job E: enterprise `1h`
- job F: SHAP batch explanation

That gives clean artifact boundaries and avoids one oversized notebook becoming
the bottleneck.

Example batch matrix:

- job A: feeder `1h`
- job B: feeder `4h`
- job C: enterprise `1h`
- job D: explanation batch

This is more controllable than one notebook doing everything at once.

## What to proxy, and what not to proxy

### Good proxy pattern

Use a local job table or folder queue:

1. GridPulse writes a training request package locally.
2. A connected helper machine uploads it to Kaggle.
3. Kaggle runs the batch job.
4. The helper machine downloads the output.
5. GridPulse imports the model artifact.

That is an asynchronous proxy.

Recommended imported artifacts:

- trained model file, for example `.pkl`
- metrics JSON
- feature importance CSV
- optional explanation summary JSON

### Bad proxy pattern

Do not make:

- `GET /forecast/*` call Kaggle directly
- your live frontend wait on Kaggle notebook completion
- GridPulse depend on Kaggle availability for normal forecasting

That would turn the forecasting runtime into a fragile remote dependency.

## How to wire this into the current repo

Use the current backend as the source of truth, then hand off batch work.

### Step 1. Build local input

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/history/generate?days=30&seed=7"
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/fingerprint/build"
```

### Step 2. Export the training payload

Use:

- `data/forecasting/ami_history.csv`
- `data/forecasting/fingerprints.csv`

### Step 3. Train on Kaggle

Recommended notebook responsibilities:

- load the CSVs
- rebuild or validate the same feature contract
- train the selected model
- compute metrics
- write artifacts to `/kaggle/working/`

Provided repo templates:

- dataset template:
  [kaggle/dataset/README.md](/home/lets-smile/Documents/PulseGrid/gridpulse/kaggle/dataset/README.md)
- training script:
  [train_forecast.py](/home/lets-smile/Documents/PulseGrid/gridpulse/kaggle/kernel/train_forecast.py)
- notebook template:
  [gridpulse_tree_forecast_template.ipynb](/home/lets-smile/Documents/PulseGrid/gridpulse/notebooks/kaggle/gridpulse_tree_forecast_template.ipynb)

### Step 4. Import the artifact locally

Copy the returned model file into:

- `data/models/forecasting/`
- or `station-data/models/offline/`

### Step 5. Serve forecasts locally

Run the normal local endpoints again. Do not change the forecast API to depend
on Kaggle at request time.

Helper scripts:

- export bundle:
  [export_training_bundle.sh](/home/lets-smile/Documents/PulseGrid/gridpulse/scripts/kaggle/export_training_bundle.sh)
- import artifacts:
  [import_artifacts.sh](/home/lets-smile/Documents/PulseGrid/gridpulse/scripts/kaggle/import_artifacts.sh)
- async workflow:
  [async_workflow.md](/home/lets-smile/Documents/PulseGrid/gridpulse/scripts/kaggle/async_workflow.md)

## Recommended implementation for this repo

For the current GridPulse codebase:

- keep live forecasting local
- use Kaggle to retrain `tree` first
- export the artifact back into `data/models/forecasting/`
- only consider Kaggle for neural models after the local tree path is stable

That recommendation is deliberate:

- `tree` already has the cleanest active API path
- SHAP support is already designed for tree models
- importing a `.pkl` tree artifact is simpler than importing a neural runtime
- this keeps the offline station path stable

If you want remote compute integrated into the repo, the right next feature is
not a live HTTP proxy. The right next feature is:

- `job export`
- `Kaggle dataset packaging`
- `kernel metadata generation`
- `artifact import`

## Source references

Kaggle CLI and metadata flow used for this recommendation:

- Kaggle CLI overview and auth:
  https://github.com/Kaggle/kaggle-cli
- CLI docs and token setup:
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
- Kernel push, status, and output flow:
  https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md
- Dataset metadata and dataset initialization flow:
  https://github.com/Kaggle/kaggle-cli/wiki/Dataset-Metadata
- Kernel metadata and kernel initialization flow:
  https://github.com/Kaggle/kaggle-cli/wiki/Kernel-Metadata
