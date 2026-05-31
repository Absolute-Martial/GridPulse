#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BACKEND_DIR="$ROOT_DIR/backend"
PYTHON_BIN="${GRIDPULSE_PYTHON_BIN:-$BACKEND_DIR/.venv/bin/python}"
STEPS="${GRIDPULSE_CONTINUOUS_STEPS:-96}"
SEED="${GRIDPULSE_CONTINUOUS_SEED:-31}"
MODEL="${GRIDPULSE_CONTINUOUS_MODEL:-tree}"
HORIZON="${GRIDPULSE_CONTINUOUS_HORIZON:-1h}"
ENTITY_TYPE="${GRIDPULSE_CONTINUOUS_ENTITY_TYPE:-feeder}"
ENTITY_ID="${GRIDPULSE_CONTINUOUS_ENTITY_ID:-FD_RES_01}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

cd "$ROOT_DIR"
PYTHONPATH="$BACKEND_DIR" "$PYTHON_BIN" -c "from app.forecasting.continuous_training import run_continuous_training_cycle; print(run_continuous_training_cycle(steps=int('$STEPS'), seed=int('$SEED'), model='$MODEL', horizon='$HORIZON', entity_type='$ENTITY_TYPE', entity_id='$ENTITY_ID'))"

"$ROOT_DIR/scripts/kaggle/export_training_bundle.sh"

echo
echo "Continuous cycle complete."
echo "Next Kaggle commands:"
echo "  kaggle datasets version -p $ROOT_DIR/kaggle/dataset -m 'continuous grid physics AMI update'"
echo "  kaggle kernels push -p $ROOT_DIR/kaggle/kernel --timeout 3600"
