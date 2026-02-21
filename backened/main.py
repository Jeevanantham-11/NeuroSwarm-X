import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from digital_twin_core import DigitalTwin


# --------------------------------------------------
# Load ERP / WMS tasks from JSON
# --------------------------------------------------
def load_tasks_from_json(path="tasks.json"):
    with open(path, "r") as f:
        data = json.load(f)

    tasks = []
    for t in data:
        tasks.append({
            "task_id": t["task_id"],
            "pickup": tuple(t["pickup"]),
            "drop": tuple(t["drop"]),
            "priority": t.get("priority", 1)
        })
    return tasks


# --------------------------------------------------
# Run Digital Twin (offline, deterministic)
# --------------------------------------------------
def run_digital_twin_demo(tasks, dt=0.2, seed=7, disturbances=True, steps=600):
    twin = DigitalTwin(dt=dt, seed=seed)
    twin.disturbances = disturbances

    # inject predefined tasks
    for t in tasks:
        twin.tasks.append(
            type("TaskObj", (), {
                "task_id": t["task_id"],
                "pickup": t["pickup"],
                "drop": t["drop"],
                "priority": t["priority"],
                "status": "QUEUED",
                "assigned_to": None
            })()
        )

    print(f"Loaded {len(tasks)} ERP/WMS tasks")

    # run simulation
    for _ in range(steps):
        twin.step()
        time.sleep(0.0)  # keep fast (no real-time delay)

    return twin.agvs


# --------------------------------------------------
# KPI Summary per AGV (CORRECT FIELDS)
# --------------------------------------------------
def summarize_agv(agv):
    a_hist = np.array(agv._a_hist)
    lat_hist = np.array(agv._lat_hist)

    jerk = np.diff(a_hist) / 0.2 if len(a_hist) > 1 else np.array([0.0])

    return {
        "Controller": agv.controller_type,
        "RMSE (m/s)": agv.rmse,
        "Energy Proxy ∫a²dt": agv.energy,
        "Jerk RMS (m/s³)": agv.jerk_rms,
        "Avg Latency (ms)": float(np.mean(lat_hist)) if len(lat_hist) else 0.0,
        "Max Latency (ms)": float(np.max(lat_hist)) if len(lat_hist) else 0.0,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":

    print("\n=== FULL DIGITAL TWIN + ERP/WMS + VALIDATION ===\n")

    seed = int(input("Enter random seed (default 7): ").strip() or "7")
    dt = float(input("Enter control dt seconds (default 0.2): ").strip() or "0.2")
    disturbances = (input("Enable disturbances? (y/n, default y): ").strip().lower() or "y").startswith("y")

    # load tasks
    tasks = load_tasks_from_json("tasks.json")

    # run twin
    agvs = run_digital_twin_demo(
        tasks,
        dt=dt,
        seed=seed,
        disturbances=disturbances,
        steps=700
    )

    # KPI table
    rows = []
    for agv in agvs:
        rows.append(summarize_agv(agv))

    df = pd.DataFrame(rows, index=[a.agv_id for a in agvs])
    print("\n=== KPI SUMMARY ===")
    print(df)

    df.to_csv("kpi_summary_full_twin.csv")

    # --------------------------------------------------
    # PID vs MPC comparison plots (fleet average)
    # --------------------------------------------------
    pid = [a for a in agvs if a.controller_type == "PID"]
    mpc = [a for a in agvs if a.controller_type == "MPC"]

    def avg(arr, attr):
        return np.mean([getattr(a, attr) for a in arr]) if arr else 0.0

    labels = ["RMSE", "Energy", "Latency", "Jerk"]
    pid_vals = [
        avg(pid, "rmse"),
        avg(pid, "energy"),
        avg(pid, "latency_avg"),
        avg(pid, "jerk_rms"),
    ]
    mpc_vals = [
        avg(mpc, "rmse"),
        avg(mpc, "energy"),
        avg(mpc, "latency_avg"),
        avg(mpc, "jerk_rms"),
    ]

    x = np.arange(len(labels))
    w = 0.35

    plt.figure(figsize=(8, 4))
    plt.bar(x - w/2, pid_vals, w, label="PID")
    plt.bar(x + w/2, mpc_vals, w, label="MPC")
    plt.xticks(x, labels)
    plt.title("PID vs MPC – Fleet KPI Comparison")
    plt.grid(axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("pid_vs_mpc_kpi.png", dpi=200)
    plt.show()

    print("\nSaved:")
    print(" - kpi_summary_full_twin.csv")
    print(" - pid_vs_mpc_kpi.png")