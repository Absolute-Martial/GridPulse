"""LightGBM grid state classifier — trained on UCI Electrical Grid Stability.

We REUSE LightGBM's standard sklearn API end-to-end. The classifier predicts
one of 4 grid states from the 12 UCI features:

    Normal   -> stable + low stress
    Stressed -> stable but margin < threshold
    Critical -> unstable
    Surplus  -> stable + low load + good renewables (proxy via tau low)

Mapping rule from UCI's two columns (`stab` numeric margin, `stabf` label):
    stabf == "unstable"           -> Critical
    stab > +0.05                  -> Surplus       (very stable, slack capacity)
    -0.01 < stab <= +0.05         -> Normal
    -0.05 <= stab <= -0.01        -> Stressed      (stable but tight)

This produces a balanced multi-class problem the same UCI features can solve.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from gridpulse.data_loader import feature_columns, load_uci_dataset

MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "classifier.pkl"
LABELS = ["Critical", "Stressed", "Normal", "Surplus"]


def _label_row(stab: float, stabf: str) -> str:
    if stabf == "unstable":
        return "Critical"
    if stab > 0.05:
        return "Surplus"
    if stab > -0.01:
        return "Normal"
    return "Stressed"


def build_labels(df: pd.DataFrame) -> pd.Series:
    return df.apply(lambda r: _label_row(float(r["stab"]), str(r["stabf"])), axis=1)


def train_classifier(force: bool = False) -> "GridStateClassifier":
    if MODEL_PATH.exists() and not force:
        return GridStateClassifier.load()

    df = load_uci_dataset()
    X = df[feature_columns()].values
    y_str = build_labels(df)
    y = y_str.map({lab: i for i, lab in enumerate(LABELS)}).values

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        objective="multiclass",
        num_class=len(LABELS),
        random_state=42,
        verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xte, yte)])
    accuracy = float((model.predict(Xte) == yte).mean())

    clf = GridStateClassifier(model=model, accuracy=accuracy)
    clf.save()
    return clf


class GridStateClassifier:
    def __init__(self, model: lgb.LGBMClassifier, accuracy: float = 0.0):
        self.model = model
        self.accuracy = accuracy
        self.feature_columns = feature_columns()

    def predict(self, features: pd.Series | np.ndarray) -> Tuple[str, dict]:
        if isinstance(features, pd.Series):
            x = features[feature_columns()].values.reshape(1, -1)
        else:
            x = np.asarray(features).reshape(1, -1)
        proba = self.model.predict_proba(x)[0]
        class_ids = [int(value) for value in np.atleast_1d(getattr(self.model, "classes_", np.arange(len(proba))))]
        winner = class_ids[int(np.argmax(proba))]
        probabilities = {label: 0.0 for label in LABELS}
        for prob, class_id in zip(proba, class_ids):
            if 0 <= class_id < len(LABELS):
                probabilities[LABELS[class_id]] = float(prob)
        return LABELS[winner], probabilities

    def save(self, path: Optional[Path] = None) -> None:
        path = path or MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "accuracy": self.accuracy}, f)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "GridStateClassifier":
        path = path or MODEL_PATH
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return cls(model=obj["model"], accuracy=obj.get("accuracy", 0.0))
