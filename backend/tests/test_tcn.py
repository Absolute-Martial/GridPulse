from pathlib import Path

import pytest

from app.simulator.sensor_generator import generate_sensor_data


def test_tcn_forecaster_is_registered() -> None:
    from app.forecasting.model_registry import get_forecaster, list_models

    assert "tcn" in list_models()
    forecaster = get_forecaster("tcn")
    assert forecaster.model_name == "tcn"


def test_tcn_module_exports_expected_types() -> None:
    from app.forecasting.tcn import TCNForecaster, WindowedTimeSeriesDataset

    assert TCNForecaster is not None
    assert WindowedTimeSeriesDataset is not None

def test_tcn_forward_pass() -> None:
    torch = pytest.importorskip("torch")
    from app.forecasting.tcn import TemporalConvNetModel

    model = TemporalConvNetModel(input_size=18, channels=(16, 16, 16, 16), kernel_size=2, dropout=0.1)
    sample = torch.randn(4, 18, 24)
    output = model(sample)

    assert tuple(output.shape) == (4, 1)


def test_dataset_window_generation() -> None:
    pytest.importorskip("torch")
    from app.forecasting.features import build_zone_feature_frame
    from app.forecasting.tcn import WindowedTimeSeriesDataset

    frame = generate_sensor_data(hours=120, interval_minutes=60, seed=7)
    feature_frame = build_zone_feature_frame(frame, zone_id="zone-1", dropna_lags=True)
    dataset = WindowedTimeSeriesDataset(feature_frame, lookback=24, horizon=6)

    assert len(dataset) > 0
    features, target = dataset[0]
    assert tuple(features.shape) == (18, 24)
    assert tuple(target.shape) == (1,)


def test_tcn_train_one_small_batch_and_save_load(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from app.forecasting.tcn import TCNForecaster

    frame = generate_sensor_data(hours=120, interval_minutes=60, seed=7)
    model_dir = tmp_path / "forecasting"
    forecaster = TCNForecaster(
        model_dir=model_dir,
        lookback=24,
        epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        patience=2,
    )

    training = forecaster.train(frame, zone_id="zone-1", horizon=24)
    prediction = forecaster.predict(frame, zone_id="zone-1", horizon=24)
    loaded = TCNForecaster(model_dir=model_dir).load(Path(training["artifact_path"]))

    assert training["artifact_path"].endswith("tcn_zone-1_24.pt")
    assert len(prediction["forecast_series"]) == 24
    assert loaded.model_name == "tcn"
