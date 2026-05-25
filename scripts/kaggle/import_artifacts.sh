#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
SOURCE_DIR="${1:-$ROOT_DIR/kaggle/output}"
TARGET_DIR="${2:-$ROOT_DIR/data/models/forecasting}"

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Missing Kaggle output directory: $SOURCE_DIR" >&2
  exit 1
fi

mkdir -p "$TARGET_DIR"

FOUND=0
for artifact in "$SOURCE_DIR"/*.pkl; do
  if [ -f "$artifact" ]; then
    cp "$artifact" "$TARGET_DIR/"
    FOUND=1
  fi
done

for metrics in "$SOURCE_DIR"/*.json; do
  if [ -f "$metrics" ]; then
    cp "$metrics" "$TARGET_DIR/"
    FOUND=1
  fi
done

if [ "$FOUND" -eq 0 ]; then
  echo "No .pkl or .json artifacts found in: $SOURCE_DIR" >&2
  exit 1
fi

echo "Imported Kaggle artifacts into: $TARGET_DIR"
