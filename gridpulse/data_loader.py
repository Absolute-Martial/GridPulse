"""Dataset and sample-input loaders for GridPulse."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
UCI_CSV = DATA_DIR / "uci_grid_stability.csv"


def load_json(path: str | Path) -> dict:
    file_path = Path(path)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    return json.loads(file_path.read_text())


def load_topology_spec(path: str | Path) -> dict:
    return load_json(path)


def load_profile_templates(path: str | Path) -> dict:
    return load_json(path)


def load_uci_dataset(force_download: bool = False) -> pd.DataFrame:
    if UCI_CSV.exists() and not force_download:
        return pd.read_csv(UCI_CSV)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        from ucimlrepo import fetch_ucirepo

        ds = fetch_ucirepo(id=471)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    except Exception:
        df = _offline_fallback(n=2000)

    df.to_csv(UCI_CSV, index=False)
    return df


def _offline_fallback(n: int = 2000) -> pd.DataFrame:
    import numpy as np

    rng = np.random.default_rng(0)
    cols = {}
    for i in range(1, 5):
        cols[f"tau{i}"] = rng.uniform(0.5, 10.0, n)
    for i in range(1, 5):
        cols[f"p{i}"] = rng.uniform(-2.0, 2.0, n)
    for i in range(1, 5):
        cols[f"g{i}"] = rng.uniform(0.05, 1.0, n)
    df = pd.DataFrame(cols)
    df["stab"] = (
        -0.05
        + 0.01 * df[[f"tau{i}" for i in range(1, 5)]].mean(axis=1)
        - 0.02 * df[[f"g{i}" for i in range(1, 5)]].mean(axis=1)
    )
    df["stabf"] = (df["stab"] < 0).map({True: "stable", False: "unstable"})
    return df


def feature_columns() -> list[str]:
    return [f"tau{i}" for i in range(1, 5)] + [f"p{i}" for i in range(1, 5)] + [f"g{i}" for i in range(1, 5)]
