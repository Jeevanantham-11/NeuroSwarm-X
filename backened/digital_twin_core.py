import time
import random
import numpy as np
from dataclasses import dataclass, field
from heapq import heappush, heappop

from pid_lib import PIDController
from mpc_lib import MPCController


# ----------------------------
# Data Models
# ----------------------------
@dataclass
class Task:
    task_id: str
    pickup: tuple[int, int]
    drop: tuple[int, int]
    priority: int = 1
    status: str = "QUEUED"       # QUEUED / ASSIGNED / PICKUP / DROPOFF / DONE
    assigned_to: str | None = None


@dataclass
class AGV:
    agv_id: str
    controller_type: str  # PID / MPC
    pos: tuple[int, int]
    x_cont: float
    y_cont: float
    v: float = 0.0
    battery: float = 1.0

    path: list[tuple[int, int]] = field(default_factory=list)
    path_index: int = 0

    current_task: str | None = None
    phase: str = "IDLE"  # IDLE / TO_PICKUP / TO_DROPOFF

    # KPI logs
    energy: float = 0.0
    jerk_rms: float = 0.0
    rmse: float = 0.0
    latency_avg: float = 0.0

    _err_hist: list = field(default_factory=list)
    _a_hist: list = field(default_factory=list)
    _lat_hist: list = field(default_factory=list)

    # Traffic control
    blocked_ticks: int = 0
    last_reserved: tuple[int, int] | None = None


# ----------------------------
# Planner A*
# ----------------------------
def astar(grid, start, goal, blocked_cells=set()):
    rows, cols = grid.shape

    def in_bounds(r, c): return 0 <= r < rows and 0 <= c < cols
    def passable(r, c):
        if (r, c) in blocked_cells:
            return False
        return grid[r, c] == 0

    def h(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    pq = []
    heappush(pq, (0, start))
    came = {start: None}
    cost = {start: 0}

    while pq:
        _, cur = heappop(pq)
        if cur == goal:
            break

        r, c = cur
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nr, nc = r+dr, c+dc
            nxt = (nr, nc)
            if in_bounds(nr, nc) and passable(nr, nc):
                new_cost = cost[cur] + 1
                if nxt not in cost or new_cost < cost[nxt]:
                    cost[nxt] = new_cost
                    pr = new_cost + h(nxt, goal)
                    heappush(pq, (pr, nxt))
                    came[nxt] = cur

    if goal not in came:
        return []

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = came[cur]
    path.reverse()
    return path


# ----------------------------
# Twin Plant dynamics
# ----------------------------
def plant_update(v, a_cmd, dt, disturbances=True):
    if not disturbances:
        load_gain = 1.0
        friction = 0.05
    else:
        load_gain = 0.7
        friction = 0.12

    a_eff = load_gain * a_cmd - friction * np.sign(v)
    v_next = v + a_eff * dt
    return float(v_next), float(a_eff)


# ----------------------------
# Reference velocity profile
# ----------------------------
def v_ref_profile(idx, total):
    frac = idx / max(1, total)
    if frac < 0.2:
        return 0.8
    elif frac < 0.8:
        return 1.5
    else:
        return 0.6


# ----------------------------
# Scheduler
# ----------------------------
def schedule_task(task, agvs):
    def manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
    best = None
    best_score = 1e9
    for agv in agvs:
        if agv.current_task is not None:
            continue
        dist = manhattan(agv.pos, task.pickup)
        score = dist + (1.0-agv.battery)*10 + task.priority*0.1
        if score < best_score:
            best_score = score
            best = agv
    return best


# ----------------------------
# Digital Twin Engine
# ----------------------------
class DigitalTwin:
    def __init__(self, dt=0.2, seed=7):
        self.dt = dt
        random.seed(seed)
        np.random.seed(seed)

        self.grid = np.zeros((12, 12), dtype=int)
        self.grid[4, 2:10] = 1
        self.grid[7, 1:9] = 1

        self.amax = 2.0
        self.jmax = 0.6
        self.disturbances = True

        # 4 AGVs
        self.agvs = [
            AGV("PID_1", "PID", (0, 0), 0.0, 0.0, battery=0.90),
            AGV("PID_2", "PID", (2, 9), 2.0, 9.0, battery=0.88),
            AGV("MPC_1", "MPC", (11, 2), 11.0, 2.0, battery=0.95),
            AGV("MPC_2", "MPC", (6, 0), 6.0, 0.0, battery=0.96),
        ]

        self.pid_ctrl = {
            "PID_1": PIDController(dt=dt, amax=self.amax),
            "PID_2": PIDController(dt=dt, amax=self.amax),
        }
        self.mpc_ctrl = {
            "MPC_1": MPCController(dt=dt, N=12, amax=self.amax, jmax=self.jmax),
            "MPC_2": MPCController(dt=dt, N=12, amax=self.amax, jmax=self.jmax),
        }

        self.logs = []
        self.tasks: list[Task] = []
        self.tick = 0

        # Traffic control
        self.reservations: dict[tuple[int, int], str] = {}   # cell -> agv_id
        self.merge_token: int = 0  # turn-based merge pointer

        # reroute parameters
        self.REROUTE_AFTER = int(15 / self.dt)  # stuck time ~15 seconds

    # ----------------------------
    # Logs
    # ----------------------------
    def add_log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")
        self.logs = self.logs[-160:]

    # ----------------------------
    # Tasks
    # ----------------------------
    def add_random_task(self):
        tid = f"T{len(self.tasks)+1}"
        pickup = (random.randint(0, 11), random.randint(0, 11))
        drop = (random.randint(0, 11), random.randint(0, 11))
        pr = random.choice([1, 2, 3])

        task = Task(tid, pickup, drop, pr)
        self.tasks.append(task)
        self.add_log(f"ERP/WMS Injected: {tid} pickup={pickup} drop={drop} pr={pr}")
        self.try_assign_tasks()

    def try_assign_tasks(self):
        for task in self.tasks:
            if task.status != "QUEUED":
                continue

            agv = schedule_task(task, self.agvs)
            if agv is None:
                continue

            # plan path to pickup + drop
            p1 = astar(self.grid, agv.pos, task.pickup)
            p2 = astar(self.grid, task.pickup, task.drop)[1:]
            path = p1 + p2

            if not path:
                task.status = "DONE"
                self.add_log(f"Validator ❌: No path found for {task.task_id}")
                continue

            agv.path = path
            agv.path_index = 0
            agv.current_task = task.task_id
            agv.phase = "TO_PICKUP"

            task.status = "ASSIGNED"
            task.assigned_to = agv.agv_id

            self.add_log(f"ATSME Scheduler -> {task.task_id} assigned to {agv.agv_id} ({agv.controller_type})")
            self.add_log(f"Shadow Validator ✅: Path locked for {agv.agv_id}")

    # ----------------------------
    # Traffic Control System
    # ----------------------------
    def _reserve_cell(self, agv: AGV, cell: tuple[int, int]) -> bool:
        """Reservation-based collision avoidance"""
        holder = self.reservations.get(cell)
        if holder is None or holder == agv.agv_id:
            self.reservations[cell] = agv.agv_id
            agv.last_reserved = cell
            return True
        return False

    def _release_cell(self, cell: tuple[int, int], agv_id: str):
        if self.reservations.get(cell) == agv_id:
            del self.reservations[cell]

    def _update_reservations(self):
        """Make sure current occupied cells remain reserved"""
        occupied = set()
        for a in self.agvs:
            occupied.add(a.pos)

        # clear invalid reservations
        kill = []
        for cell, holder in self.reservations.items():
            if cell not in occupied:
                continue
            # keep reserved if still occupied
        # no cleanup required for this simple version

    # ----------------------------
    # Auto Reroute
    # ----------------------------
    def _auto_reroute(self, agv: AGV):
        """Reroute path avoiding reserved cells (except own cell)"""
        if not agv.current_task:
            return

        t = next((x for x in self.tasks if x.task_id == agv.current_task), None)
        if not t:
            return

        # define target depending on phase
        goal = t.pickup if agv.phase == "TO_PICKUP" else t.drop

        blocked = set(self.reservations.keys())
        if agv.pos in blocked:
            blocked.remove(agv.pos)

        new_path = astar(self.grid, agv.pos, goal, blocked_cells=blocked)

        if not new_path:
            self.add_log(f"{agv.agv_id} ❌ reroute failed (no alternate path)")
            return

        agv.path = new_path
        agv.path_index = 0
        agv.blocked_ticks = 0
        self.add_log(f"{agv.agv_id} 🔄 auto-reroute success → new safe path")

    # ----------------------------
    # One AGV step
    # ----------------------------
    def step_agv(self, agv: AGV, allowed_to_move: bool):
        if not agv.path or agv.path_index >= len(agv.path):
            return

        target = agv.path[agv.path_index]

        # Reservation check
        if not allowed_to_move or not self._reserve_cell(agv, target):
            agv.blocked_ticks += 1

            # log only every few ticks to avoid spam
            if agv.blocked_ticks % int(1 / self.dt) == 0:
                self.add_log(f"{agv.agv_id} ⛔ waiting (traffic control)")

            # Auto reroute if stuck too long
            if agv.blocked_ticks > self.REROUTE_AFTER:
                self.add_log(f"{agv.agv_id} ⚠ deadlock detected → replanning")
                self._auto_reroute(agv)

            return

        # if reservation success, release old cell reservation
        self._release_cell(agv.pos, agv.agv_id)

        dx = target[0] - agv.x_cont
        dy = target[1] - agv.y_cont
        dist = (dx*dx + dy*dy) ** 0.5

        v_ref = v_ref_profile(agv.path_index, len(agv.path))
        v_meas = agv.v + np.random.normal(0, 0.03)

        # controller timing
        t0 = time.time()
        if agv.controller_type == "PID":
            ctrl = self.pid_ctrl[agv.agv_id]
            a_cmd = ctrl.compute(v_ref, v_meas)
        else:
            ctrl = self.mpc_ctrl[agv.agv_id]
            horizon = [v_ref] * ctrl.N
            a_cmd = ctrl.compute(v_meas, horizon)
        latency = (time.time() - t0) * 1000.0

        a_cmd = float(np.clip(a_cmd, -self.amax, self.amax))
        v_next, _ = plant_update(agv.v, a_cmd, self.dt, disturbances=self.disturbances)

        # move
        speed = max(0.0, min(agv.v, 2.0))
        if dist > 1e-4:
            ux, uy = dx/dist, dy/dist
            agv.x_cont += ux * speed * self.dt
            agv.y_cont += uy * speed * self.dt

        # arrival
        if abs(agv.x_cont - target[0]) < 0.15 and abs(agv.y_cont - target[1]) < 0.15:
            agv.pos = target
            agv.path_index += 1
            agv.blocked_ticks = 0  # reset stuck counter

            # task progress
            if agv.current_task:
                t = next((x for x in self.tasks if x.task_id == agv.current_task), None)
                if t:
                    if agv.phase == "TO_PICKUP" and target == t.pickup:
                        agv.phase = "TO_DROPOFF"
                        t.status = "PICKUP"
                        self.add_log(f"{agv.agv_id} 📦 picked {t.task_id}")

                    elif agv.phase == "TO_DROPOFF" and target == t.drop:
                        t.status = "DONE"
                        self.add_log(f"{agv.agv_id} ✅ delivered {t.task_id}")

                        agv.current_task = None
                        agv.phase = "IDLE"
                        agv.path = []
                        agv.path_index = 0

                        # release reservation after done
                        self._release_cell(agv.pos, agv.agv_id)

                        self.try_assign_tasks()

        agv.v = v_next

        # KPI logs
        err = v_ref - agv.v
        agv._err_hist.append(err)
        agv._a_hist.append(a_cmd)
        agv._lat_hist.append(latency)

        agv.energy += (a_cmd*a_cmd) * self.dt
        agv.rmse = float(np.sqrt(np.mean(np.array(agv._err_hist)**2)))
        agv.latency_avg = float(np.mean(agv._lat_hist))

        if len(agv._a_hist) > 2:
            jerk_arr = np.diff(np.array(agv._a_hist)) / self.dt
            agv.jerk_rms = float(np.sqrt(np.mean(jerk_arr**2)))

        agv.battery = max(0.0, agv.battery - 0.0005)

    # ----------------------------
    # Step (Turn-based merge)
    # ----------------------------
    def step(self):
        self.tick += 1

        # auto task injection
        if self.tick % int(4 / self.dt) == 0:
            if random.random() < 0.35:
                self.add_random_task()

        # TURN-BASED MERGE POLICY:
        # - only 2 AGVs are allowed to move per tick
        # - rotates each tick (like zipper merge)
        allow_count = 2
        idxs = [(self.merge_token + i) % len(self.agvs) for i in range(len(self.agvs))]
        allowed = set(idxs[:allow_count])

        for i, agv in enumerate(self.agvs):
            self.step_agv(agv, allowed_to_move=(i in allowed))

        # rotate token
        self.merge_token = (self.merge_token + 1) % len(self.agvs)

    # ----------------------------
    # Fleet KPIs
    # ----------------------------
    def _fleet_kpis(self):
        pid = [a for a in self.agvs if a.controller_type == "PID"]
        mpc = [a for a in self.agvs if a.controller_type == "MPC"]

        def avg(lst, key):
            if not lst:
                return 0.0
            return float(np.mean([getattr(x, key) for x in lst]))

        return {
            "rmse_pid": avg(pid, "rmse"),
            "rmse_mpc": avg(mpc, "rmse"),
            "energy_pid": avg(pid, "energy"),
            "energy_mpc": avg(mpc, "energy"),
            "lat_pid": avg(pid, "latency_avg"),
            "lat_mpc": avg(mpc, "latency_avg"),
            "jerk_pid": avg(pid, "jerk_rms"),
            "jerk_mpc": avg(mpc, "jerk_rms"),
        }

    # ----------------------------
    # Export state to dashboard
    # ----------------------------
    def export_state(self):
        return {
            "dt": self.dt,
            "disturbances": self.disturbances,
            "tick": self.tick,
            "grid": self.grid.tolist(),
            # 🔴🟡🟢 TRAFFIC LIGHT DATA (NEW)
            "reservations": [
                {
                    "cell": cell,      # (x, y)
                    "agv": agv_id      # who owns it
                }
                for cell, agv_id in self.reservations.items()
            ],
            "tasks": [
                {
                    "id": t.task_id,
                    "pickup": t.pickup,
                    "drop": t.drop,
                    "priority": t.priority,
                    "status": t.status,
                    "assigned_to": t.assigned_to,
                } for t in self.tasks[-25:]
            ],
            "kpis": self._fleet_kpis(),
            "agvs": [
                {
                    "id": a.agv_id,
                    "type": a.controller_type,
                    "x": a.x_cont,
                    "y": a.y_cont,
                    "battery": a.battery,
                    "task": a.current_task,
                    "phase": a.phase,
                    "rmse": a.rmse,
                    "energy": a.energy,
                    "lat": a.latency_avg,
                    "jerk": a.jerk_rms,
                    "path": a.path,
                    "path_index": a.path_index
                }
                for a in self.agvs
            ],
            "logs": self.logs
        }
