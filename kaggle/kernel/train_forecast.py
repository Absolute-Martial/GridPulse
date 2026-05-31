from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TARGET_FEATURE_COLUMNS = (
    "contracted_md_kw",
    "temperature_c",
    "humidity_percent",
    "hour",
    "minute",
    "slot_index",
    "day_of_week",
    "month",
    "season_index",
    "is_weekend",
    "is_holiday",
    "production_schedule_kw",
    "fingerprint_mean_kw",
    "fingerprint_p10_kw",
    "fingerprint_p90_kw",
    "lag_1",
    "lag_4",
    "lag_16",
    "lag_96",
    "rolling_mean_4",
    "rolling_mean_16",
    "rolling_mean_96",
)

HORIZON_TO_STEPS = {"1h": 4, "4h": 16, "24h": 96}
DEFAULT_DATASET_ROOT = Path("/kaggle/input/gridpulse-forecasting-inputs")


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    denominator = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denominator)) * 100.0)
    r2 = float(r2_score(y_true, y_pred))

    true_peak_index = int(np.argmax(y_true))
    pred_peak_index = int(np.argmax(y_pred))
    peak_time_error = abs(pred_peak_index - true_peak_index)
    peak_load_error = float(abs(float(y_pred[pred_peak_index]) - float(y_true[true_peak_index])))

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "mape": round(mape, 6),
        "r2": round(r2, 6),
        "peak_time_error": peak_time_error,
        "peak_load_error": round(peak_load_error, 6),
    }


def load_run_config() -> dict[str, object] | None:
    config_path = os.getenv(
        "GRIDPULSE_KAGGLE_RUN_CONFIG_PATH",
        str(DEFAULT_DATASET_ROOT / "run_config.json"),
    )
    resolved_path = Path(config_path)
    if not resolved_path.exists() and Path("/kaggle/input").exists():
        matches = sorted(Path("/kaggle/input").glob("*/run_config.json"))
        if matches:
            resolved_path = matches[0]
    if not resolved_path.exists():
        return None
    with resolved_path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_training_job(config: dict[str, object] | None) -> dict[str, object]:
    if config is None:
        entity_type = os.getenv("GRIDPULSE_ENTITY_TYPE", "feeder")
        entity_id = os.getenv("GRIDPULSE_ENTITY_ID", "FD_RES_01")
        horizon = os.getenv("GRIDPULSE_HORIZON", "1h").lower()
        return {
            "job_name": f"{entity_type}_{entity_id}_{horizon}",
            "dataset_file": os.getenv("GRIDPULSE_KAGGLE_HISTORY_FILE", "ami_history.csv"),
            "entity_type": entity_type,
            "entity_id": entity_id,
            "horizon": horizon,
            "output_name": f"tree_{entity_type}_{entity_id}_{horizon}",
            "fingerprints_file": os.getenv("GRIDPULSE_KAGGLE_FINGERPRINT_FILE", "fingerprints.csv"),
        }

    job_name = os.getenv("GRIDPULSE_JOB_NAME")
    jobs = config.get("jobs", [])
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("run_config.json must contain at least one job.")

    if job_name:
        for job in jobs:
            if str(job.get("job_name", "")).strip() == job_name:
                return dict(job)
        raise ValueError(f"Job '{job_name}' was not found in run_config.json.")

    return dict(jobs[0])


def load_inputs(job: dict[str, object], config: dict[str, object] | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_root = resolve_dataset_root()
    history_file = str(job.get("dataset_file", "ami_history.csv"))
    fingerprints_file = str(
        job.get(
            "fingerprints_file",
            (config or {}).get("default_fingerprints_file", "fingerprints.csv"),
        )
    )
    history_path = dataset_root / history_file
    fingerprint_path = dataset_root / fingerprints_file
    if history_path.exists():
        history = pd.read_csv(history_path)
    else:
        print(f"Training file not found at {history_path}. Generating grid-physics fallback data.")
        history = generate_grid_physics_history(entity_type=str(job["entity_type"]), days=45)

    if fingerprint_path.exists():
        fingerprints = pd.read_csv(fingerprint_path)
    else:
        fingerprints = build_fingerprints(history)

    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    return history, fingerprints


def resolve_dataset_root() -> Path:
    env_root = os.getenv("GRIDPULSE_KAGGLE_DATASET_ROOT")
    if env_root:
        return Path(env_root)
    if DEFAULT_DATASET_ROOT.exists():
        return DEFAULT_DATASET_ROOT
    if Path("/kaggle/input").exists():
        candidates = [path for path in Path("/kaggle/input").iterdir() if path.is_dir()]
        for candidate in candidates:
            if (candidate / "run_config.json").exists() or (candidate / "feeder.csv").exists():
                return candidate
    return DEFAULT_DATASET_ROOT


def build_fingerprints(history: pd.DataFrame) -> pd.DataFrame:
    return (
        history.groupby(["entity_type", "entity_id", "day_of_week", "slot_index"], as_index=False)
        .agg(
            fingerprint_mean_kw=("load_kw", "mean"),
            fingerprint_p10_kw=("load_kw", lambda series: float(series.quantile(0.10))),
            fingerprint_p90_kw=("load_kw", lambda series: float(series.quantile(0.90))),
        )
        .sort_values(["entity_type", "entity_id", "day_of_week", "slot_index"])
        .reset_index(drop=True)
    )


def generate_grid_physics_history(entity_type: str = "feeder", days: int = 45) -> pd.DataFrame:
    rng = np.random.default_rng(31)
    end = pd.Timestamp("2026-01-31T23:45:00Z")
    timestamps = pd.date_range(end=end, periods=days * 24 * 4, freq="15min", tz="UTC")
    if entity_type == "substation":
        profile = {
            "entity_type": "substation",
            "entity_id": "SS_KTM_01",
            "secondary_substation_id": "SS_KTM_01",
            "transformer_id": "TR_SS_01",
            "feeder_id": "",
            "feeder_type": "mixed",
            "customer_group_id": "",
            "customer_type": "mixed",
            "enterprise_id": "",
            "is_dedicated_line": 0,
            "contracted_md_kw": 1600.0,
            "base_load_kw": 1080.0,
            "morning_peak_kw": 120.0,
            "evening_peak_kw": 210.0,
        }
    else:
        profile = {
            "entity_type": "feeder",
            "entity_id": "FD_RES_01",
            "secondary_substation_id": "SS_KTM_01",
            "transformer_id": "TR_FD_01",
            "feeder_id": "FD_RES_01",
            "feeder_type": "residential",
            "customer_group_id": "CG_RES_01",
            "customer_type": "residential",
            "enterprise_id": "",
            "is_dedicated_line": 0,
            "contracted_md_kw": 520.0,
            "base_load_kw": 310.0,
            "morning_peak_kw": 34.0,
            "evening_peak_kw": 82.0,
        }
    records = []
    for timestamp in timestamps:
        slot_index = int(timestamp.hour * 4 + timestamp.minute // 15)
        is_weekend = int(timestamp.dayofweek >= 5)
        season, season_index = season_for_month(timestamp.month)
        hour_float = timestamp.hour + timestamp.minute / 60.0
        morning = profile["morning_peak_kw"] * np.exp(-0.5 * ((hour_float - 8.0) / 1.6) ** 2)
        evening = profile["evening_peak_kw"] * np.exp(-0.5 * ((hour_float - 19.0) / 2.2) ** 2)
        temperature_c = 18.0 + season_index * 2.0 + 5.5 * np.sin((slot_index / 96.0) * 2 * np.pi) + rng.normal(0, 0.7)
        humidity_percent = float(np.clip(58 + season_index * 5 + rng.normal(0, 4), 35, 95))
        weekend_factor = 0.94 if is_weekend else 1.0
        fingerprint_mean_kw = (profile["base_load_kw"] + morning + evening) * weekend_factor
        weather_adjustment = max(temperature_c - 24.0, 0.0) * 1.2
        load_kw = max(fingerprint_mean_kw + weather_adjustment + rng.normal(0, profile["base_load_kw"] * 0.025), 1.0)
        records.append(
            {
                "timestamp": timestamp.isoformat(),
                "entity_type": profile["entity_type"],
                "entity_id": profile["entity_id"],
                "secondary_substation_id": profile["secondary_substation_id"],
                "transformer_id": profile["transformer_id"],
                "feeder_id": profile["feeder_id"],
                "feeder_type": profile["feeder_type"],
                "customer_group_id": profile["customer_group_id"],
                "customer_type": profile["customer_type"],
                "enterprise_id": profile["enterprise_id"],
                "is_dedicated_line": profile["is_dedicated_line"],
                "contracted_md_kw": profile["contracted_md_kw"],
                "load_kw": round(float(load_kw), 3),
                "interval_energy_kwh": round(float(load_kw) * 0.25, 3),
                "temperature_c": round(float(temperature_c), 3),
                "humidity_percent": round(float(humidity_percent), 3),
                "day_type": "weekend" if is_weekend else "weekday",
                "hour": timestamp.hour,
                "minute": timestamp.minute,
                "slot_index": slot_index,
                "month": timestamp.month,
                "day_of_week": timestamp.dayofweek,
                "season": season,
                "season_index": season_index,
                "is_weekend": is_weekend,
                "is_holiday": 0,
                "data_quality_flag": "synthetic_ok",
                "source_type": "synthetic_ami",
                "production_schedule_kw": 0.0,
                "fingerprint_mean_kw": round(float(fingerprint_mean_kw), 3),
                "fingerprint_p10_kw": round(float(fingerprint_mean_kw) * 0.92, 3),
                "fingerprint_p90_kw": round(float(fingerprint_mean_kw) * 1.08, 3),
            }
        )
    return pd.DataFrame.from_records(records)


def season_for_month(month: int) -> tuple[str, int]:
    if month in (12, 1, 2):
        return "winter", 0
    if month in (3, 4):
        return "spring", 1
    if month in (5, 6):
        return "summer", 2
    if month in (7, 8, 9):
        return "monsoon", 3
    return "autumn", 4


def filter_target_history(history: pd.DataFrame, entity_type: str, entity_id: str) -> pd.DataFrame:
    filtered = history[
        (history["entity_type"].astype(str).str.lower() == entity_type.lower())
        & (history["entity_id"].astype(str) == entity_id)
    ].copy()
    if filtered.empty:
        raise ValueError(f"Unknown target: {entity_type} {entity_id}")
    return filtered.sort_values("timestamp").reset_index(drop=True)


def build_feature_frame(history: pd.DataFrame) -> pd.DataFrame:
    working = history.copy()
    shifted_load = working["load_kw"].shift(1)
    working["lag_1"] = working["load_kw"].shift(1)
    working["lag_4"] = working["load_kw"].shift(4)
    working["lag_16"] = working["load_kw"].shift(16)
    working["lag_96"] = working["load_kw"].shift(96)
    working["rolling_mean_4"] = shifted_load.rolling(4, min_periods=1).mean()
    working["rolling_mean_16"] = shifted_load.rolling(16, min_periods=1).mean()
    working["rolling_mean_96"] = shifted_load.rolling(96, min_periods=1).mean()
    return working


def build_training_dataset(feature_frame: pd.DataFrame, horizon_steps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base = feature_frame.dropna(subset=["lag_1", "lag_4", "lag_16", "lag_96"]).reset_index(drop=True)
    for index in range(len(base) - horizon_steps):
        target_sequence = base["load_kw"].iloc[index + 1 : index + 1 + horizon_steps].astype(float).tolist()
        row = base.iloc[index].to_dict()
        row["target_sequence"] = target_sequence
        rows.append(row)
    if not rows:
        raise ValueError("No target windows were generated. Increase history length.")
    return pd.DataFrame(rows)


def train_tree_artifact(
    history: pd.DataFrame,
    entity_type: str,
    entity_id: str,
    horizon: str,
    n_estimators: int = 300,
    min_samples_leaf: int = 2,
    random_state: int = 7,
) -> tuple[dict[str, object], dict[str, object]]:
    horizon_steps = HORIZON_TO_STEPS[horizon]
    target_history = filter_target_history(history, entity_type, entity_id)
    feature_frame = build_feature_frame(target_history)
    dataset = build_training_dataset(feature_frame, horizon_steps=horizon_steps)

    split_index = max(1, int(len(dataset) * 0.8))
    if split_index >= len(dataset):
        split_index = len(dataset) - 1

    train_frame = dataset.iloc[:split_index].copy()
    test_frame = dataset.iloc[split_index:].copy()

    x_train = train_frame.loc[:, list(TARGET_FEATURE_COLUMNS)]
    y_train = np.vstack(train_frame["target_sequence"].to_list())
    x_test = test_frame.loc[:, list(TARGET_FEATURE_COLUMNS)]
    y_test = np.vstack(test_frame["target_sequence"].to_list())

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    metrics = calculate_metrics(y_test.flatten(), predictions.flatten())
    residuals = y_test - predictions
    lower_residual = float(np.quantile(residuals, 0.10))
    upper_residual = float(np.quantile(residuals, 0.90))

    artifact = {
        "model_name": "tree",
        "entity_type": entity_type,
        "entity_id": entity_id,
        "horizon": horizon,
        "horizon_steps": horizon_steps,
        "feature_columns": list(TARGET_FEATURE_COLUMNS),
        "model": model,
        "metrics": metrics,
        "lower_residual": lower_residual,
        "upper_residual": upper_residual,
    }
    summary = {
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "metrics": metrics,
    }
    return artifact, summary


def main() -> None:
    config = load_run_config()
    job = resolve_training_job(config)
    history, _fingerprints = load_inputs(job, config)
    entity_type = str(job.get("entity_type", "feeder"))
    entity_id = str(job.get("entity_id", "FD_RES_01"))
    horizon = str(job.get("horizon", "1h")).lower()
    if horizon not in HORIZON_TO_STEPS:
        raise ValueError(f"Unsupported horizon: {horizon}")

    artifact, summary = train_tree_artifact(
        history=history,
        entity_type=entity_type,
        entity_id=entity_id,
        horizon=horizon,
    )

    output_dir = Path(os.getenv("GRIDPULSE_KAGGLE_OUTPUT_DIR", "/kaggle/working"))
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_entity_id = entity_id.replace("/", "_")
    output_name = str(job.get("output_name", f"tree_{entity_type}_{safe_entity_id}_{horizon}"))
    artifact_path = output_dir / f"{output_name}.pkl"
    metrics_path = output_dir / f"{output_name}.json"

    with artifact_path.open("wb") as artifact_file:
        pickle.dump(artifact, artifact_file)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump({"job": job, **summary}, metrics_file, indent=2)

    print(json.dumps({"job": job, "artifact_path": str(artifact_path), "metrics_path": str(metrics_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
