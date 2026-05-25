#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
DATASET_DIR="${1:-$ROOT_DIR/kaggle/dataset}"
AMI_SOURCE="${GRIDPULSE_AMI_HISTORY_PATH:-$ROOT_DIR/data/forecasting/ami_history.csv}"
FINGERPRINT_SOURCE="${GRIDPULSE_FORECAST_FINGERPRINT_PATH:-$ROOT_DIR/data/forecasting/fingerprints.csv}"

mkdir -p "$DATASET_DIR"

if [ ! -f "$AMI_SOURCE" ]; then
  echo "Missing AMI history file: $AMI_SOURCE" >&2
  exit 1
fi

if [ ! -f "$FINGERPRINT_SOURCE" ]; then
  echo "Missing fingerprint file: $FINGERPRINT_SOURCE" >&2
  exit 1
fi

cp "$AMI_SOURCE" "$DATASET_DIR/ami_history.csv"
cp "$FINGERPRINT_SOURCE" "$DATASET_DIR/fingerprints.csv"

awk -F',' 'NR==1 || $2=="feeder"' "$AMI_SOURCE" > "$DATASET_DIR/feeder.csv"
awk -F',' 'NR==1 || $2=="substation"' "$AMI_SOURCE" > "$DATASET_DIR/substation.csv"

echo "Exported Kaggle training bundle to: $DATASET_DIR"
echo "Files:"
echo "  - $DATASET_DIR/ami_history.csv"
echo "  - $DATASET_DIR/feeder.csv"
echo "  - $DATASET_DIR/substation.csv"
echo "  - $DATASET_DIR/fingerprints.csv"
echo
echo "Next steps:"
echo "  1. kaggle datasets version -p $DATASET_DIR -m 'updated forecasting inputs'"
echo "  2. kaggle kernels push -p $ROOT_DIR/kaggle/kernel --timeout 3600"
