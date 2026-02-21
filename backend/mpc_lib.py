import numpy as np

class MPCController:
    """
    Lightweight MPC-like controller (predictive rollout + choose best accel)
    No cvxpy dependency -> runs anywhere.
    """
    def __init__(self, dt=0.2, N=12, amax=2.0, jmax=0.6,
                 w_track=160.0, w_accel=2.2, w_jerk=18.0):
        self.dt = dt
        self.N = N
        self.amax = amax
        self.jmax = jmax
        self.w_track = w_track
        self.w_accel = w_accel
        self.w_jerk = w_jerk
        self.prev_a = 0.0

    def compute(self, v0, vref_horizon):
        # candidate accel samples
        candidates = np.linspace(-self.amax, self.amax, 21)

        best_cost = 1e18
        best_a = 0.0

        for a0 in candidates:
            # jerk constraint from previous
            if abs(a0 - self.prev_a) > self.jmax:
                continue

            v = v0
            a_prev = self.prev_a
            cost = 0.0
            a = a0

            for k in range(self.N):
                # dynamics
                v = v + a * self.dt

                # tracking
                cost += self.w_track * (v - vref_horizon[k])**2
                cost += self.w_accel * (a**2)
                cost += self.w_jerk * ((a - a_prev)**2)

                # next step accel decay to smooth (simple)
                a_prev = a
                a = np.clip(a*0.85, -self.amax, self.amax)

            if cost < best_cost:
                best_cost = cost
                best_a = a0

        self.prev_a = best_a
        return float(best_a)
