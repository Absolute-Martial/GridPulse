from app.forecasting.fingerprint_baseline import FingerprintBaselineForecaster
from app.forecasting.feeder_fingerprint import build_fingerprint_database
from app.forecasting.metrics import calculate_regression_metrics
from app.forecasting.tree_forecaster import TreeForecaster
from app.simulator.ami_history_generator import generate_ami_history


def test_fingerprint_baseline_predicts_slot_sequence(tmp_path) -> None:
    history = generate_ami_history(days=14, seed=7)
    fingerprint = build_fingerprint_database(history)
    model = FingerprintBaselineForecaster(model_dir=tmp_path)
    model.fit_fingerprint(fingerprint)
    model.artifact.update(  # type: ignore[union-attr]
        {
            "entity_type": "feeder",
            "entity_id": "FD_RES_01",
            "horizon": "1h",
        }
    )
    prediction = model.predict(
        history,
        entity_type="feeder",
        entity_id="FD_RES_01",
        horizon="1h",
    )

    assert prediction["horizon"] == "1h"
    assert len(prediction["slot_predictions"]) == 4


def test_tree_forecaster_trains_and_predicts_target_slots(tmp_path) -> None:
    history = generate_ami_history(days=21, seed=7)
    forecaster = TreeForecaster(model_dir=tmp_path)
    train_result = forecaster.train(
        history,
        entity_type="feeder",
        entity_id="FD_RES_01",
        horizon="4h",
    )
    prediction = forecaster.predict(
        history,
        entity_type="feeder",
        entity_id="FD_RES_01",
        horizon="4h",
    )

    assert train_result["artifact_path"].endswith("tree_feeder_FD_RES_01_4h.pkl")
    assert len(prediction["slot_predictions"]) == 16
    assert prediction["summary"]["confidence"] in {"high", "medium", "low"}


def test_metrics_include_peak_errors() -> None:
    metrics = calculate_regression_metrics(
        actual=[100.0, 120.0, 180.0, 140.0],
        predicted=[98.0, 118.0, 170.0, 145.0],
    )

    assert {"mae", "rmse", "mape", "r2", "peak_time_error", "peak_load_error"} <= set(metrics)
