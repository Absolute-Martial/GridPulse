# GridPulse Kaggle Dataset Template

This folder is the upload template for Kaggle dataset input used by the current
GridPulse forecasting MVP.

Expected upload files:

- `feeder.csv`
- `substation.csv`
- `fingerprints.csv`
- `run_config.json`
- `dataset-metadata.json`

Included examples:

- `sample_ami_history.csv`
- `sample_feeder.csv`
- `sample_substation.csv`
- `sample_fingerprints.csv`

## Canonical AMI history schema

Base files:

- `feeder.csv`
- `substation.csv`
- optional combined `ami_history.csv`

Column types:

- `timestamp`: ISO-8601 UTC datetime string
- `entity_type`: string
- `entity_id`: string
- `secondary_substation_id`: string
- `transformer_id`: string
- `feeder_id`: string
- `feeder_type`: string
- `customer_group_id`: string
- `customer_type`: string
- `enterprise_id`: string, nullable for non-enterprise rows
- `is_dedicated_line`: integer `0/1`
- `contracted_md_kw`: float
- `load_kw`: float
- `interval_energy_kwh`: float
- `temperature_c`: float
- `humidity_percent`: float
- `day_type`: string
- `hour`: integer
- `minute`: integer
- `slot_index`: integer in `[0, 95]`
- `month`: integer in `[1, 12]`
- `day_of_week`: integer in `[0, 6]`
- `season`: string
- `season_index`: integer
- `is_weekend`: integer `0/1`
- `is_holiday`: integer `0/1`
- `data_quality_flag`: string
- `source_type`: string
- `production_schedule_kw`: float
- `fingerprint_mean_kw`: float
- `fingerprint_p10_kw`: float
- `fingerprint_p90_kw`: float

Important rule:

- if raw AMI data is provided as `interval_energy_kwh`, then
  `load_kw = interval_energy_kwh / 0.25`

## Fingerprint schema

Base file: `fingerprints.csv`

Column types:

- `entity_type`: string
- `entity_id`: string
- `day_of_week`: integer
- `slot_index`: integer
- `fingerprint_mean_kw`: float
- `fingerprint_p10_kw`: float
- `fingerprint_p90_kw`: float

## Suggested dataset contents

For practical Kaggle training, upload:

- the target-specific `feeder.csv`
- the target-specific `substation.csv`
- the matching `fingerprints.csv`
- the matching `run_config.json`

## run_config.json

Use `run_config.json` to tell the Kaggle script which dataset file and target
to train.

Required job fields:

- `job_name`
- `dataset_file`
- `entity_type`
- `entity_id`
- `horizon`
- `output_name`
