# ADAS Module

## Overview
The ADAS module integrates Adaptive Cruise Control (ACC) and Lane Keeping (LK) functionality using CARLA simulator, Kuksa Databroker, and Zenoh.

## Files
- **Env.py** : Publishes CARLA camera via Zenoh and radar-derived ACC inputs to Kuksa.
- **decision.py** : Computes LK steering (camera) + ACC control (radar) and publishes 
