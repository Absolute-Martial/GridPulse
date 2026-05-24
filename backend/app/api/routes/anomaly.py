"""Anomaly detection routes."""

from fastapi import APIRouter, HTTPException

from app.anomaly.anomaly_detector import (
    AnomalyModelNotFoundError,
    AnomalyTrainingError,
    GridAnomalyDetector,
)
from app.synthetic.csv_adapter import CsvSchemaError, load_sensor_frame

router = APIRouter(tags=["anomaly"])


@router.post("/anomaly/train")
async def train_anomaly_detector() -> dict:
    frame = _load_frame_or_raise()
    detector = GridAnomalyDetector()
    try:
        result = detector.train(frame)
    except AnomalyTrainingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "ok",
        **result,
    }


@router.get("/anomaly/latest")
async def latest_anomalies() -> dict:
    frame = _load_frame_or_raise()
    detector = GridAnomalyDetector()
    try:
        anomalies = detector.detect_latest(frame)
    except AnomalyModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "ok",
        "count": len(anomalies),
        "anomalies": anomalies,
    }


@router.get("/anomaly/history")
async def anomaly_history() -> dict:
    frame = _load_frame_or_raise()
    detector = GridAnomalyDetector()
    try:
        anomalies = detector.detect_history(frame)
    except AnomalyModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "status": "ok",
        "count": len(anomalies),
        "anomalies": anomalies,
    }


def _load_frame_or_raise():
    try:
        frame = load_sensor_frame()
    except CsvSchemaError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if frame is None:
        raise HTTPException(status_code=400, detail="No synthetic sensor history is available.")
    return frame
