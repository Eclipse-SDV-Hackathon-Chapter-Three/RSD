# Project Execution Guide

This guide explains how to run the project step by step.
Make sure you have Ankaios, CARLA Simulator, and the proper Conda environment installed. (There is python version problem with CARLA simulator)

## 1. Start Ankaios
Apply the system state configuration :
```bash
ank set state desiredState /home/yg/ankaios/state.yaml
```

## 2. Launch CARLA Simulator
Run CARLA with custom settings (low quality, windowed mode, 1280*720) :
```bash
./CarlaUE4.sh -carla-rpc-port=2000 -quality-level=Low -windowed -ResX=1280 -ResY=720
```

## 3. Activate Conda Environment
Activate the Conda environment prepared for CARLA :
```bash
conda activate carla-0915
```

### Conda Environment Setup
This project requires a Conda environment (`carla-0915`) with CARLA, Kuksa, Zenoh, and other dependencies.

---

#### 1. Create Conda Environment
Create and activate the environment with Python 3.10:
```bash
conda create -n carla-0915 python=3.10
conda activate carla-0915
```

#### 2. Install Required Packages
Install the necessary packages using pip inside the Conda environment:
```bash
# Core simulation
pip install carla==0.9.15
pip install pygame

# Middleware & communication
pip install eclipse-zenoh==1.5.1
pip install kuksa-client==0.4.0
pip install grpcio grpcio-tools websockets

# Data processing
pip install numpy pandas opencv-python pillow

# Utilities
pip install requests jsonschema gitpython
```

## 4. Run ADAS Modules
Inside the activated Conda environments, run the modules in order:
1. Env.py — Publishes CARLA sensor data (camera/radar) to Kuksa/Zenoh
```bash
python Env.py
```

2. decision.py — Runs Lane Keeping + Adaptive Cruise Control logic
```bash
python decision.py
```

3. control.py — Applies throttle, brake, and steering commands to the CARLA ego vehicle
```bash
python control.py
```

4. Run Test Scenario
```bash
python lead_scenario.py
```

## Notes
- Ensure zenoh.json5 and state.yaml are properly configured before starting.
- The order of execution is important: Ankaios → CARLA → Conda Env → ADAS modules → Scenario.
