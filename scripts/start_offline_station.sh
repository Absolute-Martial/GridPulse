#!/usr/bin/env sh

set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
DATA_DIR="$ROOT_DIR/station-data"

mkdir -p "$DATA_DIR/synthetic"
mkdir -p "$DATA_DIR/processed"
mkdir -p "$DATA_DIR/models/offline"

cd "$ROOT_DIR"
docker compose up --build
