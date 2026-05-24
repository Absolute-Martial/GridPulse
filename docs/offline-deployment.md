# Offline Deployment

GridPulse is packaged for station-side deployment with no internet dependency
after the initial image build and dependency download.

## Services

- `backend`: FastAPI API on port `8000`
- `frontend`: local dashboard on port `8080`
- `storage`: local file-storage helper that keeps the station data directories present

## One-Time Install With Internet

Run this once on a connected machine or on the target station before it is
air-gapped:

```bash
cd gridpulse
docker compose up --build
```

What this first run does:

- pulls the base container images
- installs Python dependencies into the backend image
- builds the frontend image
- creates the local station data directories

After that first build, GridPulse can be started again without internet access
as long as the built images remain on the station.

## Run Offline

From the repository root:

```bash
cd gridpulse
docker compose up --build
```

Or use the startup script:

```bash
cd gridpulse
sh scripts/start_offline_station.sh
```

The runtime stack does not call cloud APIs and does not require network access
for normal operation.

## Runtime Data Storage

GridPulse stores runtime files in the host-mounted directory:

`gridpulse/station-data/`

Mapped paths inside the backend container:

- host `gridpulse/station-data/synthetic/`
  container `/app/data/synthetic/`
- host `gridpulse/station-data/processed/`
  container `/app/data/processed/`
- host `gridpulse/station-data/models/`
  container `/app/data/models/`

Main files:

- AMI history CSV:
  `gridpulse/station-data/synthetic/ami_history.csv`
- fingerprint database CSV:
  `gridpulse/station-data/processed/fingerprints.csv`
- forecasting artifacts:
  `gridpulse/station-data/models/forecasting/`

You can also keep manually supplied offline model artifacts in:

`gridpulse/station-data/models/offline/`

## Backup Sensor CSV Files

To back up all sensor CSV files:

```bash
cd gridpulse
mkdir -p backups
cp -r station-data/synthetic backups/synthetic-$(date +%Y%m%d-%H%M%S)
```

To back up the full station dataset:

```bash
cd gridpulse
mkdir -p backups
cp -r station-data backups/station-data-$(date +%Y%m%d-%H%M%S)
```

## Load New Model Files Manually

If you receive a new forecasting artifact from a trusted offline source:

1. Stop the stack if you want a clean swap:

```bash
cd gridpulse
docker compose down
```

2. Copy the artifact into the station model directory:

```bash
mkdir -p gridpulse/station-data/models/forecasting
cp /path/to/tree_feeder_FD_RES_01_1h.pkl gridpulse/station-data/models/forecasting/
```

3. Optionally keep the original drop in the offline archive folder:

```bash
cp /path/to/tree_feeder_FD_RES_01_1h.pkl gridpulse/station-data/models/offline/
```

4. Start the stack again:

```bash
cd gridpulse
docker compose up --build
```

## Operational Notes

- The host-mounted `station-data/` directory is the source of truth for runtime
  data on the station.
- The backend image ships with seeded `data/` content, but the bind-mounted
  `station-data/` directory overrides it at runtime.
- If `station-data/synthetic/ami_history.csv` is missing, generate fresh AMI
  history through the forecasting history endpoint after startup.
