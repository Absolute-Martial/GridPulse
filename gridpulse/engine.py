"""High-level orchestration layer for the GridPulse demo."""
from __future__ import annotations

from gridpulse.anomaly import AnomalyDetector
from gridpulse.classifier import GridStateClassifier, train_classifier
from gridpulse.config import load_settings
from gridpulse.data_loader import load_profile_templates
from gridpulse.forecasting import ForecastModel
from gridpulse.optimizer import generate_dispatch_plan
from gridpulse.schemas import EngineSnapshot, GridStatePrediction, Settings
from gridpulse.simulator import Simulator
from gridpulse.suggestions import generate_suggestions
from gridpulse.topology import grid_summary, load_demo_grid


class GridPulseEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        self.graph = load_demo_grid(self.settings.grid.topology_path)
        self.profiles = load_profile_templates(self.settings.grid.profile_path)
        topology_nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            node = dict(attrs)
            node["id"] = node_id
            topology_nodes.append(node)
        self.forecaster = ForecastModel.train(self.profiles, {"nodes": topology_nodes}, seed=self.settings.app.random_seed)
        self.classifier: GridStateClassifier = train_classifier()
        self.anomaly = AnomalyDetector()
        self.simulator = Simulator(
            graph=self.graph,
            profiles=self.profiles,
            forecaster=self.forecaster,
            seed=self.settings.app.random_seed,
            default_scenario=self.settings.scenario.default,
        )

    def set_scenario(self, name: str) -> None:
        self.simulator.set_scenario(name)

    def inject_fault(self, node_id: str) -> None:
        self.simulator.inject_fault(node_id)

    def clear_fault(self, node_id: str) -> None:
        self.simulator.clear_fault(node_id)

    def grid_summary(self) -> dict:
        return grid_summary(self.graph)

    def advance_tick(self) -> EngineSnapshot:
        timestamp, scenario, zone_states, features = self.simulator.advance()
        label, probabilities = self.classifier.predict(features)
        anomaly_score = self.anomaly.update(features)
        for zone in zone_states:
            zone.anomaly_score = anomaly_score
        prediction = GridStatePrediction(label=label, probabilities=probabilities)
        dispatch = generate_dispatch_plan(
            graph=self.graph,
            zone_states=zone_states,
            faulted_nodes=list(self.simulator.faulted_nodes),
            enable_pandapower=self.settings.validation.enable_pandapower,
        )
        suggestions = generate_suggestions(
            classifier=self.classifier,
            features=features,
            prediction_label=prediction.label,
            probabilities=prediction.probabilities,
            anomaly_score=anomaly_score,
            graph=self.graph,
            zone_states=zone_states,
            dispatch=dispatch,
            faulted_nodes=list(self.simulator.faulted_nodes),
            scenario_name=scenario.name,
        )
        return EngineSnapshot(
            tick=self.simulator.tick - 1,
            timestamp=timestamp,
            scenario=scenario,
            zone_states=zone_states,
            prediction=prediction,
            dispatch=dispatch,
            suggestions=suggestions,
            anomaly_score=anomaly_score,
            faulted_nodes=list(self.simulator.faulted_nodes),
        )
