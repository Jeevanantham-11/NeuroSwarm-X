# NeuroSwarm-X  
### Digital Twin–Driven Multi-AGV Coordination (PID vs MPC)

## Overview
NeuroSwarm-X is a **digital twin simulation system** for warehouse AGVs that compares **PID** and **MPC controllers** under congestion, task scheduling, and disturbances.  
It supports **collision-free traffic control**, **ERP/WMS task inputs**, and a **live dashboard**.

## Features
- 4 AGVs (2 PID + 2 MPC)
- Reservation-based traffic lights (no collisions)
- Turn-based merge + auto reroute
- Live KPIs (RMSE, Energy, Latency, Jerk)
- Offline KPI plots and CSV export

---

## How to Run

### 1. Install dependencies
```
pip install numpy pandas matplotlib fastapi uvicorn
```
### 2. Run backend (Digital Twin Server)
```
cd backened
uvicorn server:app --reload
```
```

Backend runs at:

http://127.0.0.1:8000
```
### 3. Open Dashboard (index.html)
Option A: Using VS Code Live Server (Recommended)

Open project folder in VS Code

Right-click index.html

Click “Open with Live Server”

Dashboard opens automatically in browser

Option B: Open directly (basic)

Double-click index.html
(Live updates may be limited without Live Server)

### 4. Run Offline Simulation (KPIs + plots)
```
python main.py
```
Outputs:

kpi_summary_full_twin.csv

full_velocity.png

full_jerk.png

full_energy.png

full_latency.png

### Notes

Tasks are loaded from tasks.json

Dashboard updates in real-time via WebSocket

Traffic lights show reserved grid cells (collision prevention)