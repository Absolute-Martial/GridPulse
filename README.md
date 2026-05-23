# GridPulse

GridPulse is a real-time smart-grid decision-support system for forecasting, optimization, routing, anomaly detection, and explainable recommendations.

## Built with

Python, FastAPI, Streamlit, NetworkX, scikit-learn, LightGBM, SHAP, pandapower, pandas, NumPy, Matplotlib, Uvicorn, uv, GitHub.

## Contribution

GridPulse introduces a hybrid AI-driven smart grid framework that integrates:

- demand forecasting
- multi-objective optimization
- graph-based routing
- anomaly detection
- explainable AI

into a unified, real-time operator decision support system.

This integration is the core novelty of the project, distinguishing it from single-purpose approaches.

## Research Motivation

Modern smart grids face increasing complexity due to renewable energy integration, fluctuating demand, and distributed generation. These challenges introduce:

- non-linear and unpredictable demand patterns
- conflicting optimization objectives across cost, reliability, and sustainability
- limited adaptability in static routing systems
- lack of interpretability in AI-driven decisions
- high vulnerability to anomalies and faults in real-time operations

Traditional grid management systems address these challenges in isolation through forecasting, optimization, or routing, but they do not provide an integrated, real-time, operator-centric solution.

GridPulse addresses this gap by combining forecasting, optimization, routing, anomaly detection, and explainable AI into a unified decision-support system.

## Hybrid Architecture

GridPulse implements a multi-layer hybrid architecture combining machine learning, optimization, and graph-based reasoning:

### 1. Forecasting Layer
- Predicts near-term demand using time-series features
- Current: regression baseline (`scikit-learn`)
- Future: LSTM / Temporal Fusion Transformer

### 2. Optimization Layer
- Multi-objective load balancing
- Objectives:
  - minimize line loss
  - minimize cost
  - reduce overload risk
  - maximize renewable utilization
- Solver: linear programming + graph cost heuristics

### 3. Graph Routing Layer
- Grid modeled as a graph with nodes for substations and edges for lines
- Routing:
  - shortest path (`Dijkstra`)
  - min-cost flow dispatch
- Fault-aware rerouting

### 4. Anomaly Detection Layer
- Isolation Forest detects:
  - abnormal load spikes
  - instability patterns
  - abnormal system states

### 5. Explainable Decision Layer
- LightGBM predicts grid stability
- SHAP explains:
  - why recommendations are generated
  - feature-level contribution to system state

### 6. Recommendation Engine
- Generates ranked operator actions:
  - shift load
  - isolate faults
  - reroute energy
  - reduce demand

## System Pipeline

GridPulse processes grid data through the following pipeline:

Input:
- historical load data
- synthetic renewable generation
- network topology
- scenario (`normal`, `peak`, `fault`)

Pipeline:
1. Demand forecasting -> predicted load per zone
2. Optimization -> balanced dispatch plan
3. Routing -> power flow through graph
4. Anomaly detection -> system health scoring
5. Classification -> grid stability probability
6. SHAP explanation -> feature importance
7. Suggestion engine -> ranked operator actions

Output:
- updated grid state
- recommendations with confidence
- explanation traces for operator interpretation

## Experimental Scenarios

GridPulse simulates an 8-zone smart grid under multiple operating conditions:

- normal operation
- peak demand conditions
- renewable surplus scenarios
- fault injection and recovery

### Observed Capabilities

- adaptive load redistribution under stress
- real-time fault isolation and rerouting
- anomaly detection in unstable grid states
- explainable recommendations for operator action
- confidence-ranked decision support outputs

## Research Basis

GridPulse is inspired by recent advances in smart grid research:

- LSTM-based load forecasting for time-series modeling
- multi-objective optimization for grid balancing
- graph-based routing for energy distribution
- isolation forest for anomaly detection in high-dimensional data
- SHAP-based explainable AI for interpretable decision-making

Unlike existing approaches, GridPulse integrates all components into a single real-time system.

## Backend API

The backend service runs at:

```bash
uvicorn api.main:app --reload --port 8010
```

Key endpoints:

- `GET /health`
- `GET /summary`
- `POST /scenario/{name}`
- `POST /faults/{node_id}`
- `DELETE /faults/{node_id}`
- `POST /tick`
- `GET /snapshot?scenarios=normal&scenarios=fault`
- `POST /integrations/snapshot/export?scenarios=normal&scenarios=fault`

## Documentation

- Setup instructions: [setup instruction.md](setup%20instruction.md)
- Demo flow: [docs/demo_script.md](docs/demo_script.md)
- Architecture notes: [docs/architecture.md](docs/architecture.md)

## Limitations

- Uses simplified graph-based load flow instead of full AC power flow analysis
- Forecasting model is baseline and not deep learning yet
- Synthetic datasets instead of real-world SCADA data
- No real-time distributed streaming integration
- Decision policies are heuristic rather than learned policies

## Future Work

- integrate deep learning forecasting (LSTM / TFT)
- incorporate reinforcement learning for control policies
- extend to AC power flow models using pandapower
- integrate real-time telemetry (SCADA, IoT)
- evaluate against baseline systems (with vs without GridPulse)
- deploy as a digital twin for smart grid simulation
