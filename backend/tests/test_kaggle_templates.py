from __future__ import annotations

import csv
import json
from pathlib import Path

from app.forecasting.schema import CANONICAL_HISTORY_COLUMNS, FINGERPRINT_COLUMNS


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_kaggle_sample_ami_history_has_canonical_headers() -> None:
    sample_path = REPO_ROOT / "kaggle" / "dataset" / "sample_ami_history.csv"
    assert sample_path.exists()

    with sample_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

    assert tuple(header) == CANONICAL_HISTORY_COLUMNS


def test_kaggle_sample_fingerprints_have_expected_headers() -> None:
    sample_path = REPO_ROOT / "kaggle" / "dataset" / "sample_fingerprints.csv"
    assert sample_path.exists()

    with sample_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        header = next(reader)

    assert header[:4] == ["entity_type", "entity_id", "day_of_week", "slot_index"]
    assert tuple(header[-3:]) == FINGERPRINT_COLUMNS


def test_kaggle_kernel_metadata_template_has_required_fields() -> None:
    metadata_path = REPO_ROOT / "kaggle" / "kernel" / "kernel-metadata.json"
    assert metadata_path.exists()

    payload = json.loads(metadata_path.read_text())

    for required_field in (
        "id",
        "title",
        "code_file",
        "language",
        "kernel_type",
        "is_private",
        "enable_gpu",
        "enable_internet",
        "dataset_sources",
    ):
        assert required_field in payload


def test_kaggle_notebook_template_exists() -> None:
    notebook_path = REPO_ROOT / "notebooks" / "kaggle" / "gridpulse_tree_forecast_template.ipynb"
    assert notebook_path.exists()


def test_kaggle_run_config_exists_and_has_jobs() -> None:
    config_path = REPO_ROOT / "kaggle" / "dataset" / "run_config.json"
    assert config_path.exists()

    payload = json.loads(config_path.read_text())
    assert "jobs" in payload
    assert isinstance(payload["jobs"], list)
    assert payload["jobs"]


def test_kaggle_split_sample_datasets_exist() -> None:
    feeder_path = REPO_ROOT / "kaggle" / "dataset" / "sample_feeder.csv"
    substation_path = REPO_ROOT / "kaggle" / "dataset" / "sample_substation.csv"

    assert feeder_path.exists()
    assert substation_path.exists()
