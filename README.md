# RSD - Ready Set Drive !

<p align="center">
  <img src="https://contrib.rocks/image?repo=Eclipse-SDV-Hackathon-Chapter-Three/RSD" alt="Contributors" />
</p>

## Team members
- Gihoon Kim
- Kijin Baek
- Changseok Oh
- Younggyo Oh
- Siwoo Lee

---
## Challenge : Virtual SDV Lab Challenge

---
## Core Idea
### Problem Definition
ADAS features such as ACC, AEB, and LKAS are becoming increasingly important for vehicle safety. However, their performance can vary depending on road surface condition such as wet, icy, snowy, or dry roads. We wanted to enhance ADAS safety and reliability by incorporating real-time road condition awareness into the system.

### Rough Solution Idea
Our approach is to develop a **Road Condition Prediction Layer** that estimates road surface condition and classifies road surface types using multi-sensor data (e.g. camera, radar, vehicle dynamics).

These layers are then integrated with ADAS features to refine thier behavior:
- **ACC(Adaptive Cruise Control)** : ACC automatically maintains a set speed chosen by the driver, while also adjusting the vehicle's speed to keep a safe distance from the car ahead.
- **AEB(Automatic Emergency Braking)** : AEB detects the risk of a collision with vehicles, pedestrains, or obstacles and automatically applies the brakes in unavoidable situations to prevent the accident.
- **LKAS(Lane Keeping Assist System)** : LKAS prevents the vehicle from unintentionally departing its current lane by providing gentle steering corrections.

By integrating road condition prediction with ADAS, our system aims to achieve safer and more adaptive driving across diverse conditions.

### System Architecture
<img width="1602" height="755" alt="Screenshot from 2025-09-30 19-07-35" src="https://github.com/user-attachments/assets/e71f12de-2452-449a-a66c-08073fa596c1" />


---
## Project Workflow
### Development process
- Design project architecture.
- Based on project architecture, each team member will be responsible for developing specific modules.
- Communication between modules will be aligned and refined through continuous integration.
- Appropriate Eclipse tools will be applied in each part of the project to support data communication and orchestration.
- CARLA simulator will be used regularly to validate the system, identify errors, and improve reliability through iterative testing and debugging.
- Project progress and decisions will be continuously documented in the shared workspace, allowing all team members to access and collaborate effectively.
- Decisions will be made collaboratively through open discussions, ensuring the project moves in the right direction during the hackathon.



  
