import sys

from app.explainability.shap_forecast_explainer import explain_tree_forecast
from app.forecasting.tree_forecaster import TreeForecaster
from app.simulator.ami_history_generator import generate_ami_history


def test_explain_tree_forecast_falls_back_when_shap_is_unavailable(tmp_path, monkeypatch) -> None:
    history = generate_ami_history(days=21, seed=7)
    forecaster = TreeForecaster(model_dir=tmp_path)
    forecaster.train(history, entity_type="feeder", entity_id="FD_RES_01", horizon="1h")
    monkeypatch.setitem(sys.modules, "shap", None)

    explanation = explain_tree_forecast(
        forecaster=forecaster,
        dataframe=history,
        entity_type="feeder",
        entity_id="FD_RES_01",
        horizon="1h",
    )

    assert explanation["target"] == "peak"
    assert explanation["method"] in {"shap", "permutation_importance"}
    assert explanation["top_features"]
    assert explanation["plain_language_explanation"]
