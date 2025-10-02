# Common Module

## Overview
The common module provides shared algorithms, signal specifications, and configuration files for ADAS and RCR modules.

## Files
- **LK_algo.py** : Lane detection and steering gain helpers.
- **acc_algo.py** : Adaptive Cruise Control core logic with road-aware thresholds.
- **myvss.json** : Extended VSS tree definition (ACC, LK, Road, Slip, Fused State).
- **state.yaml** : Initial VSS state values.
- **zenoh.json5** : Zenoh router configuration.

## Key Features
- Lane Keeping : memory-based lane fusion, dynamic lookahead ratio
- ACC : Cruise / Follow / AEB states, hysteresis (delayed switching) by road state
- Shared Configs : VSS schema (myvss.json), state overlays (state.yaml), transport optimization (zenoh.json5)

## Notes
- 'myvss.json' must be loaded into Kuksa Databroker before execution.
- 'zenoh.json5' configures Zenoh router with SHM (Shared Memory) enabled.
