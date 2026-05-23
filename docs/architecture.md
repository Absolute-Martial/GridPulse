# GridPulse Architecture

## Scope

GridPulse is built as a hackathon-first operator decision-support prototype. The default path optimizes for reliability, local setup speed, and demo clarity rather than electrical-grade infrastructure.

## Runtime pipeline

```text
Sample topology + sample load profiles
        ->
Scenario simulator
        ->
Forecast baseline (RandomForestRegressor)
        ->
Grid-state classifier (LightGBM)
        ->
Anomaly detector (Isolation Forest)
        ->
Dispatch optimizer (NetworkX min-cost flow)
        ->
Suggestion engine (rules + SHAP)
        ->
Streamlit operator dashboard
```

## Module map

- `gridpulse/config.py`
  - loads `config/defaults.yaml`
  - defines the single main configuration surface
- `gridpulse/topology.py`
  - loads the committed sample topology
  - builds the NetworkX graph
  - exposes optional pandapower validation
- `gridpulse/simulator.py`
  - generates zone-level loads from scenario presets and daily demand profiles
  - creates UCI-style feature vectors for ML modules
- `gridpulse/forecasting.py`
  - trains a small scikit-learn forecast baseline from committed profile data
- `gridpulse/classifier.py`
  - trains and caches the LightGBM grid-state classifier
- `gridpulse/anomaly.py`
  - maintains the Isolation Forest anomaly score
- `gridpulse/optimizer.py`
  - computes served demand, flow utilization, and route metrics
- `gridpulse/suggestions.py`
  - turns system state into ranked operator actions and SHAP-supported explanations
- `gridpulse/engine.py`
  - orchestrates one full tick of the pipeline

## Open-source reuse strategy

- `OpenEMS`
  - documentation baseline only
  - used to compare “operational EMS” vs “explainable AI co-pilot”
- `NetworkX`
  - default routing and min-cost flow engine
- `pandapower`
  - optional validation path
  - not required for the main demo path
- `UCI Electrical Grid Stability`
  - classifier training reference
- `NREL/OpenEI`
  - represented here through compact local profile samples to keep setup easy

## Why NetworkX-first

The current MVP defaults to NetworkX because it keeps the prototype easy to configure, fast to explain, and robust for hackathon demos. Full electrical modeling is valuable, but making it the default path adds setup and demo risk. The repository keeps `pandapower` available for validation and future upgrades without forcing that complexity into the core demo loop.
