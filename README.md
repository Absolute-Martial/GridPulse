# GridPulse

GridPulse is an offline smart-grid forecasting prototype. The current
repository now centers on canonical 15-minute AMI history, target-aware load
forecasting, and tree-based forecast explanation, with a Docker-served web
dashboard.

## Current active scope

- Offline-first backend with no cloud APIs
- Canonical 15-minute AMI history generation
- Fingerprint database builder
- Target-aware forecasting for substations, feeders, and enterprise lines
- `fingerprint_baseline` and `tree` forecast models
- Tree forecast explanation with SHAP fallback
- Docker-served static forecasting dashboard

Inactive for this phase:

- Anomaly detection
- OPF and pandapower validation
- GNN risk scoring
- Routing and control actions
- Recommendation and allocation flows

## Local backend setup

```bash
cd gridpulse/backend
env UV_CACHE_DIR=/tmp/uv-cache uv venv .venv
env UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt --python .venv/bin/python
env UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.tcn.txt --python .venv/bin/python
```

## Rebuild the backend environment

```bash
cd gridpulse/backend
rm -rf .venv
uv venv --clear .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -r requirements.tcn.txt
```

## Run the backend

```bash
cd gridpulse/backend
.venv/bin/python -m uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## Run the Docker deployment

```bash
cd gridpulse
docker compose up --build
```

## GitHub image build

The repository includes [docker-image.yml](/home/lets-smile/Documents/PulseGrid/gridpulse/.github/workflows/docker-image.yml).

- Pull requests: build and validate backend and frontend images
- `main` and `v*` tags: build and push images to `ghcr.io/<owner>/gridpulse-backend` and `ghcr.io/<owner>/gridpulse-frontend`
- The workflow installs base backend dependencies from `requirements.txt` and TCN/Torch dependencies from `requirements.tcn.txt`

Services:

- Backend API: `http://127.0.0.1:8000`
- Dashboard: `http://127.0.0.1:8080`

## Current forecasting endpoints

`POST /api/v1/forecast/history/generate`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/history/generate?days=30&seed=7"
```

`POST /api/v1/forecast/fingerprint/build`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/fingerprint/build"
```

`POST /api/v1/forecast/train`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/forecast/train?feeder_id=FD_RES_01&horizon=1h&model=tree"
```

`GET /api/v1/forecast/feeder`

```bash
curl "http://127.0.0.1:8000/api/v1/forecast/feeder?feeder_id=FD_RES_01&horizon=1h&model=tree"
```

`GET /api/v1/forecast/evaluate`

```bash
curl "http://127.0.0.1:8000/api/v1/forecast/evaluate?feeder_id=FD_RES_01&horizon=1h&model=tree"
```

`GET /api/v1/explain/forecast`

```bash
curl "http://127.0.0.1:8000/api/v1/explain/forecast?feeder_id=FD_RES_01&horizon=1h&model=tree"
```

## Legacy and auxiliary endpoints

`GET /health`

```bash
curl http://127.0.0.1:8000/health
```

`GET /api/v1/sensors/latest`

```bash
curl http://127.0.0.1:8000/api/v1/sensors/latest
```

`GET /api/v1/sensors/history?hours=24`

```bash
curl "http://127.0.0.1:8000/api/v1/sensors/history?hours=24"
```

`POST /api/v1/simulation/generate`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/simulation/generate?hours=24&interval_minutes=60&seed=7"
```

`GET /api/v1/grid/state`

```bash
curl http://127.0.0.1:8000/api/v1/grid/state
```

`GET /api/v1/grid/summary`

```bash
curl http://127.0.0.1:8000/api/v1/grid/summary
```

`GET /api/v1/grid/overloads`

```bash
curl http://127.0.0.1:8000/api/v1/grid/overloads
```

`POST /api/v1/grid/fault/inject`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/grid/fault/inject?line_id=line-2"
```

`POST /api/v1/grid/fault/clear`

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/grid/fault/clear?line_id=line-2"
```

The current forecasting storage paths are:

```bash
export GRIDPULSE_AMI_HISTORY_PATH=/absolute/path/to/ami_history.csv
export GRIDPULSE_FORECAST_FINGERPRINT_PATH=/absolute/path/to/fingerprints.csv
export GRIDPULSE_FORECAST_ARTIFACT_DIR=/absolute/path/to/forecasting-artifacts
```

Additional docs:

- [architecture.md](/home/lets-smile/Documents/PulseGrid/gridpulse/docs/architecture.md)
- [api-spec.md](/home/lets-smile/Documents/PulseGrid/gridpulse/docs/api-spec.md)
- [data-schema.md](/home/lets-smile/Documents/PulseGrid/gridpulse/docs/data-schema.md)
- [kaggle-training.md](/home/lets-smile/Documents/PulseGrid/gridpulse/docs/kaggle-training.md)

## Run tests

```bash
cd gridpulse/backend
.venv/bin/python -m pytest -q
```
