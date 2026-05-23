# References

## Open-source tools and baselines

- OpenEMS
  - https://openems.github.io/openems.io/openems/latest/introduction.html
  - https://github.com/OpenEMS/openems
- NetworkX flow algorithms
  - https://networkx.org/documentation/stable/reference/algorithms/flow.html
- pandapower
  - https://www.pandapower.org/
  - https://github.com/e2nIEE/pandapower
- SHAP LightGBM example
  - https://shap-community.readthedocs.io/en/latest/example_notebooks/tabular_examples/tree_based_models/Census%20income%20classification%20with%20LightGBM.html

## Datasets

- UCI Electrical Grid Stability Simulated Data
  - https://archive.ics.uci.edu/dataset/471/electrical+grid+stability+simulated+data
- OpenEI / NREL End-Use Load Profiles
  - https://data.openei.org/submissions/4520
  - https://registry.opendata.aws/nrel-pds-building-stock/

## Positioning note

OpenEMS is used here as a comparison baseline for real EMS architecture. GridPulse does not depend on OpenEMS at runtime. The current repository focuses on a simpler hackathon-ready prototype with a NetworkX-default graph model and optional pandapower validation.
