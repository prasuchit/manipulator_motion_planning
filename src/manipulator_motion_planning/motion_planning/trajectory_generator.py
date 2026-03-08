
from numpy.typing import NDArray

class CubicTrajectory:
    def __init__(self, 
                 start_pos: NDArray,
                 end_pos: NDArray,
                 start_vel: NDArray,
                 end_vel: NDArray,
                 duration: float,
                 ):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.start_vel = start_vel
        self.end_vel = end_vel
        self.duration = duration
        self.compute_coefficients()

    def compute_coefficients(self):
        # Simple renaming to help with math which I did on paper
        # Don't worry if you don't get this, Quite easy to derive
        # from this: 
        # p(t) = a + bt + ct^2 + dt^3
        # v(t) = b + 2ct + 3dt^2
        # and applying boundary conditions

        p0 = self.start_pos
        pT = self.end_pos
        v0 = self.start_vel
        vT = self.end_vel
        T = self.duration

        self.a = p0
        self.b = v0

        # Intermediate values
        A = pT - p0 - v0 * T 
        B = vT - v0

        self.d = (B*T - 2*A) / (T**3)
        self.c = (B - 3*self.d*(T**2)) / (2*T)

    def get_position(self, t):
        # p(t) = a + bt + ct^2 + dt^3
        return self.a + self.b * t + self.c * (t**2) + self.d * (t**3)
    
    def get_velocity(self, t):
        # v(t) = b + 2ct + 3dt^2
        return self.b + 2 * self.c * t + 3 * self.d * (t**2)
    
    def get_acceleration(self, t):
        # a(t) = 2c + 6dt
        return 2 * self.c + 6 * self.d * t

    def get_duration(self):
        return self.duration