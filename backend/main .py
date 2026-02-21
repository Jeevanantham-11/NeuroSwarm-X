import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backened.digital_twin_core import load_tasks_from_json, run_digital_twin_demo


def summarize_agv(agv):
    t = np.array(agv.t_log)
    v = np.array(agv.v_log)
    a = np.array(agv.a_log)
    j = np.array(agv.jerk_log)
    e = np.array(agv.energy_log)
    lat = np.array(agv.latency_log)

    return {
        "Avg Velocity (m/s)": float(np.mean(v)) if len(v) else 0.0,
        "Peak Jerk (m/s^3)": float(np.max(np.abs(j))) if len(j) else 0.0,
        "Jerk RMS (m/s^3)": float(np.sqrt(np.mean(j*j))) if len(j) else 0.0,
        "Energy Proxy ∫a²dt": float(e[-1]) if len(e) else 0.0,
        "Avg Latency (ms)": float(np.mean(lat)) if len(lat) else 0.0,
        "Max Latency (ms)": float(np.max(lat)) if len(lat) else 0.0,
    }


if __name__ == "__main__":
    print("\n=== FULL DIGITAL TWIN + ERP/WMS + VALIDATOR DEMO ===\n")

    # Dynamic inputs
    seed = int(input("Enter random seed (default 7): ").strip() or "7")
    dt = float(input("Enter control dt seconds (default 0.2): ").strip() or "0.2")
    disturbances = (input("Enable disturbances? (y/n, default y): ").strip().lower() or "y").startswith("y")

    # Load ERP/WMS tasks
    tasks = load_tasks_from_json("tasks.json")

    # Run digital twin
    agvs = run_digital_twin_demo(tasks, dt=dt, seed=seed, disturbances=disturbances)

    # KPI / Validation summary
    rows = []
    for agv in agvs:
        rows.append((agv.agv_id, agv.controller_type, summarize_agv(agv)))

    df = pd.DataFrame([r[2] for r in rows], index=[f"{r[0]} ({r[1]})" for r in rows])
    print("\n=== KPI SUMMARY ===")
    print(df)
    df.to_csv("kpi_summary_full_twin.csv")

    # Comparison plots
    pid = [a for a in agvs if a.controller_type == "PID"][0]
    mpc = [a for a in agvs if a.controller_type == "MPC"][0]

    plt.figure()
    plt.plot(pid.t_log, pid.v_log, label="PID velocity")
    plt.plot(mpc.t_log, mpc.v_log, label="MPC velocity")
    plt.title("Velocity (PID vs MPC)")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid(True)
    plt.legend()
    plt.savefig("full_velocity.png", dpi=200)

    plt.figure()
    plt.plot(pid.t_log, pid.jerk_log, label="PID jerk")
    plt.plot(mpc.t_log, mpc.jerk_log, label="MPC jerk")
    plt.title("Jerk (PID vs MPC)")
    plt.xlabel("Time (s)")
    plt.ylabel("Jerk (m/s^3)")
    plt.grid(True)
    plt.legend()
    plt.savefig("full_jerk.png", dpi=200)

    plt.figure()
    plt.plot(pid.t_log, pid.energy_log, label="PID energy proxy")
    plt.plot(mpc.t_log, mpc.energy_log, label="MPC energy proxy")
    plt.title("Energy Proxy ∫a²dt (PID vs MPC)")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy proxy")
    plt.grid(True)
    plt.legend()
    plt.savefig("full_energy.png", dpi=200)

    plt.figure()
    plt.plot(pid.t_log, pid.latency_log, label="PID latency")
    plt.plot(mpc.t_log, mpc.latency_log, label="MPC latency")
    plt.title("Latency (PID vs MPC)")
    plt.xlabel("Time (s)")
    plt.ylabel("Latency (ms)")
    plt.grid(True)
    plt.legend()
    plt.savefig("full_latency.png", dpi=200)

    plt.show()

    print("\nSaved: kpi_summary_full_twin.csv")
    print("Saved plots: full_velocity.png, full_jerk.png, full_energy.png, full_latency.png")
