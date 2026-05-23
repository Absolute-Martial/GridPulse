# GridPulse Setup Instructions

## Prerequisites

- Python 3.11+
- `uv` recommended for dependency management

## 1. Enter the repo

```bash
cd /home/lets-smile/OpsRadar/gridpulse-project
```

## 2. Create the virtual environment

```bash
uv venv .venv
source .venv/bin/activate
```

If `uv` cache permissions are restricted in your environment:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv venv --clear .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

If needed, keep the cache on `/tmp`:

```bash
env UV_CACHE_DIR=/tmp/uv-cache uv pip install -r requirements.txt
```

## 4. Run the backend API

```bash
uvicorn api.main:app --reload --port 8010
```

Backend checks:

- `http://127.0.0.1:8010/health`
- `http://127.0.0.1:8010/summary`

## 5. Run the frontend prototype

In a second terminal:

```bash
streamlit run dashboard/app.py
```

If you want an explicit port:

```bash
streamlit run dashboard/app.py --server.headless true --server.port 8501
```

Frontend check:

- `http://127.0.0.1:8501`

## 6. Run the tests

```bash
PYTHONPATH=. ./.venv/bin/pytest -q
```

## 7. Export a local snapshot

The backend can export a JSON snapshot for local integration demos:

```bash
python scripts/export_openems_snapshot.py
```

That writes to:

- `artifacts/gridpulse-snapshot.json`
