"""Lightweight demand forecasting built on scikit-learn."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class ForecastModel:
    model: RandomForestRegressor
    zone_order: list[str]

    @classmethod
    def train(cls, profiles: dict, topology_spec: dict, seed: int = 7) -> "ForecastModel":
        rows: list[dict] = []
        zone_base = {node["id"]: float(node.get("base_load_kw", 0.0)) for node in topology_spec["nodes"] if node["kind"] == "zone"}
        temperatures = profiles["temperature_c"]
        solar = profiles["solar_factor"]
        wind = profiles["wind_factor"]

        for zone_id, series in profiles["zones"].items():
            base = zone_base[zone_id]
            for hour, multiplier in enumerate(series):
                next_hour = (hour + 1) % len(series)
                next_multiplier = series[next_hour]
                for scenario_code, scenario_factor in enumerate([1.0, 1.18, 0.92]):
                    current_load = base * multiplier * scenario_factor
                    target_load = base * next_multiplier * scenario_factor
                    rows.append(
                        {
                            "zone_id": zone_id,
                            "hour": hour,
                            "current_load_kw": current_load,
                            "temperature_c": temperatures[hour],
                            "solar_factor": solar[hour],
                            "wind_factor": wind[hour],
                            "scenario_code": scenario_code,
                            "target_load_kw": target_load,
                        }
                    )

        frame = pd.DataFrame(rows)
        features = pd.get_dummies(frame[["zone_id", "hour", "current_load_kw", "temperature_c", "solar_factor", "wind_factor", "scenario_code"]], columns=["zone_id"])
        model = RandomForestRegressor(n_estimators=120, random_state=seed)
        model.fit(features, frame["target_load_kw"])
        return cls(model=model, zone_order=sorted(profiles["zones"].keys()))

    def predict_zone_load(self, zone_id: str, hour: int, current_load_kw: float, temperature_c: float, solar_factor: float, wind_factor: float, scenario_code: int) -> float:
        row = pd.DataFrame(
            [
                {
                    "zone_id": zone_id,
                    "hour": hour,
                    "current_load_kw": current_load_kw,
                    "temperature_c": temperature_c,
                    "solar_factor": solar_factor,
                    "wind_factor": wind_factor,
                    "scenario_code": scenario_code,
                }
            ]
        )
        features = pd.get_dummies(row, columns=["zone_id"])
        for zone_key in self.zone_order:
            col = f"zone_id_{zone_key}"
            if col not in features.columns:
                features[col] = 0
        features = features.reindex(sorted(features.columns), axis=1)

        trained_columns = getattr(self.model, "feature_names_in_", None)
        if trained_columns is not None:
            features = features.reindex(trained_columns, axis=1, fill_value=0)
        return float(self.model.predict(features)[0])
