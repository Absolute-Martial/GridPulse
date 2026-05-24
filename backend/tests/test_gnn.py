from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.simulator.sensor_generator import generate_sensor_data, save_sensor_data


def test_graph_tensor_creation() -> None:
    from app.gnn.graph_dataset import build_graph_tensors

    frame = generate_sensor_data(hours=48, interval_minutes=60, seed=7)
    sample = build_graph_tensors(frame)

    assert sample.node_features.ndim == 2
    assert sample.edge_index.shape[0] == 2
    assert sample.edge_features.ndim == 2
    assert sample.node_targets.ndim == 2
    assert sample.edge_targets.ndim == 2
    assert sample.metadata["node_ids"]
    assert sample.metadata["edge_ids"]


def test_gnn_forward_pass() -> None:
    from app.gnn.graph_dataset import build_graph_tensors
    from app.gnn.models import GridRiskGNN

    frame = generate_sensor_data(hours=48, interval_minutes=60, seed=7)
    sample = build_graph_tensors(frame)
    model = GridRiskGNN(
        node_feature_dim=sample.node_features.size(1),
        edge_feature_dim=sample.edge_features.size(1),
        hidden_dim=32,
        num_layers=2,
    )
    output = model(sample.node_features, sample.edge_index, sample.edge_features)

    assert output.node_risk_score.shape[0] == sample.node_features.shape[0]
    assert output.line_risk_score.shape[0] == sample.edge_features.shape[0]
    assert output.overload_risk.shape[0] == sample.edge_features.shape[0]
    assert output.restoration_priority.shape[0] == sample.node_features.shape[0]


def test_gnn_training_one_small_epoch(tmp_path: Path) -> None:
    from app.gnn.trainer import GridRiskGNNTrainer

    frame = generate_sensor_data(hours=48, interval_minutes=60, seed=7)
    trainer = GridRiskGNNTrainer(model_path=tmp_path / "grid_risk_gnn.pt")
    result = trainer.train(frame, epochs=1, scenarios_per_type=1, hidden_dim=32, num_layers=2)

    assert result["artifact_path"].endswith("grid_risk_gnn.pt")
    assert result["train_graphs"] > 0
    assert (tmp_path / "grid_risk_gnn.pt").exists()


def test_gnn_inference_output_format(tmp_path: Path) -> None:
    from app.gnn.inference import GridRiskInference
    from app.gnn.trainer import GridRiskGNNTrainer

    frame = generate_sensor_data(hours=48, interval_minutes=60, seed=7)
    model_path = tmp_path / "grid_risk_gnn.pt"
    GridRiskGNNTrainer(model_path=model_path).train(frame, epochs=1, scenarios_per_type=1, hidden_dim=32, num_layers=2)

    inference = GridRiskInference(model_path=model_path)
    risks = inference.predict_risk(frame)

    assert risks
    assert {
        "risk_score",
        "risk_type",
        "affected_node",
        "affected_line",
        "confidence",
        "features_used",
    } <= set(risks[0])


@pytest.mark.anyio
async def test_gnn_api_endpoints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "sensor_data.csv"
    model_path = tmp_path / "grid_risk_gnn.pt"
    frame = generate_sensor_data(hours=48, interval_minutes=60, seed=7)
    save_sensor_data(frame, csv_path=csv_path)
    monkeypatch.setenv("GRIDPULSE_SYNTHETIC_CSV_PATH", str(csv_path))
    monkeypatch.setenv("GRIDPULSE_GNN_MODEL_PATH", str(model_path))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
        train_response = await client.post("/api/v1/gnn/train?epochs=1&scenarios_per_type=1")
        predict_response = await client.get("/api/v1/gnn/predict-risk")
        lines_response = await client.get("/api/v1/gnn/top-risk-lines?limit=3")
        zones_response = await client.get("/api/v1/gnn/top-risk-zones?limit=3")

    assert train_response.status_code == 200
    assert predict_response.status_code == 200
    assert lines_response.status_code == 200
    assert zones_response.status_code == 200
    assert predict_response.json()["count"] > 0
