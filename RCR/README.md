# RCR Module (Road Condition Recognition)

## Overview
The RCR module estimates road surface conditions using camera-based analysis and dynamics-based slip estimation, then fuses them into a final state.

## Files
- **kuksa_road_analyzer.py** : Classifies road state from camera data (dry / wet / snow / icy).
- **kuksa_slip_estimator.py** : Estimates slip state from IMU, wheel slip, and odometry sensors.
- **kuksa_state_fuser.py** : Fuses camera and slip states, publishes 'StateFused'.

## Data Flow
- Camera (Zenoh) → Road Analyzer → Kuksa
- Ego dynamics (IMU, wheel slip) → Slip Estimator → Kuksa
- State Fuser → Combines both (camera and slip) → Kuksa ('Vehicle.Private.StateFused.*')

## Key Features
- Metrics : Specular Reflection Index (SRI, aka. reflection rate), Laplacian variance (aka. clarity), Edge density (aka. how many patterns, edges)
- Slip scoring : lateral / longitudinal residuals, wheel slip ratio
- Fusion : timestamp matching and weighted confidence scoring

## Outputs (Kuksa custom VSS signals)
- Vehicle.Private.Road.State , Vehicle.Private.Road.Confidence , Vehicle.Private.Road.Metrics
- Vehicle.Private.Slip.State , Vehicle.Private.Slip.Confidence , Vehicle.Private.Slip.Quality
- Vehicle.Private.StateFused.State , Vehicle.Private.StateFused.Confidence
