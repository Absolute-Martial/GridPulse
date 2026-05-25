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
        "/kaggle/input/gridpulse-forecasting-inputs/run_config.json",
    )
    resolved_path = Path(config_path)
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
    dataset_root = Path(
        os.getenv(
            "GRIDPULSE_KAGGLE_DATASET_ROOT",
            "/kaggle/input/gridpulse-forecasting-inputs",
        )
    )
    history_file = str(job.get("dataset_file", "ami_history.csv"))
    fingerprints_file = str(
        job.get(
            "fingerprints_file",
            (config or {}).get("default_fingerprints_file", "fingerprints.csv"),
        )
    )
    history = pd.read_csv(dataset_root / history_file)
    fingerprints = pd.read_csv(dataset_root / fingerprints_file)
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    return history, fingerprints


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

    output_dir = Path("/kaggle/working")
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
