from app.forecasting.response_formatter import build_forecast_response


def test_build_forecast_response_uses_aggregated_summary_and_energy_rule() -> None:
    payload = build_forecast_response(
        entity_type="feeder",
        entity_id="FD_RES_01",
        horizon="1h",
        latest_timestamp="2026-05-25T10:15:00Z",
        slot_predictions=[
            {"timestamp": "2026-05-25T10:30:00Z", "predicted_load_kw": 100.0, "p10_kw": 90.0, "p90_kw": 110.0, "fingerprint_mean_kw": 98.0},
            {"timestamp": "2026-05-25T10:45:00Z", "predicted_load_kw": 120.0, "p10_kw": 100.0, "p90_kw": 130.0, "fingerprint_mean_kw": 118.0},
            {"timestamp": "2026-05-25T11:00:00Z", "predicted_load_kw": 140.0, "p10_kw": 120.0, "p90_kw": 150.0, "fingerprint_mean_kw": 138.0},
            {"timestamp": "2026-05-25T11:15:00Z", "predicted_load_kw": 160.0, "p10_kw": 150.0, "p90_kw": 170.0, "fingerprint_mean_kw": 158.0},
        ],
        confidence="medium",
        model_name="tree",
    )

    assert payload["horizon_steps"] == 4
    assert "aggregated_summary" in payload
    assert payload["summary"]["total_energy_kwh"] == 130.0
    assert payload["summary"]["peak_time"] == "2026-05-25T11:15:00Z"
