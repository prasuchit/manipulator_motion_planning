from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


@dataclass
class DriverStatus:
    time_s: float
    current_joint_positions: NDArray
    current_joint_velocities: NDArray
    current_joint_accelerations: NDArray

    @staticmethod
    def from_dict(json_data):
        return DriverStatus(
            time_s=json_data["time_s"],
            current_joint_positions=np.array(json_data["current_joint_positions"]),
            current_joint_velocities=np.array(json_data["current_joint_velocities"]),
            current_joint_accelerations=np.array(
                json_data["current_joint_accelerations"]
            ),
        )

    def to_dict(self):
        return {
            "time_s": self.time_s,
            "current_joint_positions": list(self.current_joint_positions),
            "current_joint_velocities": list(self.current_joint_velocities),
            "current_joint_accelerations": list(self.current_joint_accelerations),
        }


@dataclass
class DriverCommand:
    target_joint_positions: NDArray

    @staticmethod
    def from_dict(json_data):
        return DriverCommand(target_joint_positions=json_data["target_joint_positions"])

    def to_dict(self):
        return {"target_joint_positions": list(self.target_joint_positions)}


@dataclass
class ControllerStatus:
    joint_positions: NDArray


class Trajectory(Protocol):
    def get_position(self, t): ...
    def get_velocity(self, t): ...
    def get_acceleration(self, t): ...
    def get_duration(self): ...


@dataclass
class ControllerCommand:
    trajectory: Trajectory
