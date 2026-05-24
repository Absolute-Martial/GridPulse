# GridPulse Data Schema

## Canonical AMI history

The current forecasting system uses one canonical 15-minute schema.

Required fields:

- `timestamp`
- `entity_type`
- `entity_id`
- `secondary_substation_id`
- `transformer_id`
- `feeder_id`
- `feeder_type`
- `customer_group_id`
- `customer_type`
- `enterprise_id`
- `is_dedicated_line`
- `contracted_md_kw`
- `load_kw`
- `interval_energy_kwh`
- `temperature_c`
- `humidity_percent`
- `day_type`
- `hour`
- `minute`
- `slot_index`
- `month`
- `day_of_week`
- `season`
- `season_index`
- `is_weekend`
- `is_holiday`
- `data_quality_flag`
- `source_type`
- `production_schedule_kw`
- `fingerprint_mean_kw`
- `fingerprint_p10_kw`
- `fingerprint_p90_kw`

## Conversion rule

If `load_kw` is missing and `interval_energy_kwh` exists:

`load_kw = interval_energy_kwh / 0.25`

## Season mapping

- winter: December, January, February -> `0`
- spring: March, April -> `1`
- summer: May, June -> `2`
- monsoon: July, August, September -> `3`
- autumn: October, November -> `4`

## Fingerprint grouping keys

The fingerprint database is built by:

- `entity_type`
- `entity_id`
- `day_of_week`
- `slot_index`

Stored values:

- `fingerprint_mean_kw`
- `fingerprint_p10_kw`
- `fingerprint_p90_kw`
