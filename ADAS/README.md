# ADAS Module

## Overview
The ADAS module integrates Adaptive Cruise Control (ACC) and Lane Keeping (LK) functionality using CARLA simulator, Kuksa Databroker, and Zenoh.

## Files
- **Env.py** : Publishes CARLA camera via Zenoh and radar-derived ACC inputs to Kuksa.
- **decision.py** : Computes LK steering (camera) + ACC control (radar) and publishes results to Kuksa.
- **control.py** : Reads Kuksa control commands and applies throttle, brake, and steering to CARLA ego vehicle.

## Data Flow
Carla (Camera, Radar, Ego speed) → Env.py
- Zenoh : Camera data → decision.py → Kuksa (LK Steering)
- Kuksa : ACC sensor values (Radar) → decision.py (ACC algo) → Kuksa (Throttle / Brake / Mode selection) → control.py → CARLA Ego vehicle

## Key Features
- Lane detection with Canny + Hough transform
- Road-aware ACC controller with Cruise / Follow / AEB states
- Steering pipeline with EMA, I-term, and rate limiter

## Outputs (Kuksa custom VSS signals)
- Vehicle.ADAS.LK.Steering
- Vehicle.ADAS.ACC.Ctrl.Throttle
- Vehicle.ADAS.ACC.Ctrl.Brake
- Vehicle.ADAS.ACC.Ctrl.Mode
