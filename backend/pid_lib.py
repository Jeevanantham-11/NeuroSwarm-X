import numpy as np

class PIDController:
    def __init__(self, Kp=2.0, Ki=0.35, Kd=0.12, dt=0.2, amax=2.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.amax = amax
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, v_ref, v_meas):
        error = v_ref - v_meas
        self.integral += error * self.dt
        derivative = (error - self.prev_error) / self.dt

        a = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        return float(np.clip(a, -self.amax, self.amax))
