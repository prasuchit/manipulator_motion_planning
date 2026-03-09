from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray


@dataclass
class DriverStatus:
    time_s: float
    current_joint_positions: NDArray
    current_joint_velocities: NDArray
    current_joint_accelerations: NDArray

    @staticmethod
    def from_dict(json_data: Dict[str, Any]) -> "DriverStatus":
        return DriverStatus(
            time_s=json_data["time_s"],
            current_joint_positions=np.array(json_data["current_joint_positions"]),
            current_joint_velocities=np.array(json_data["current_joint_velocities"]),
            current_joint_accelerations=np.array(
                json_data["current_joint_accelerations"]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(json_data: Dict[str, Any]) -> "DriverCommand":
        return DriverCommand(target_joint_positions=json_data["target_joint_positions"])

    def to_dict(self) -> Dict[str, List]:
        return {"target_joint_positions": list(self.target_joint_positions)}


@dataclass
class DriverConfig:
    host: str = "localhost"
    pub_port: int = 5555
    sub_port: int = 5556
    robot_cmd_topic: str = "robot_cmd"
    robot_status_topic: str = "robot_status"

    @staticmethod
    def load_from_yaml(yaml_file: Dict[str, Any], driver_type: str) -> "DriverConfig":
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        return DriverConfig(
            host=data[driver_type]["host"],
            pub_port=data[driver_type]["pub_port"],
            sub_port=data[driver_type]["sub_port"],
            robot_cmd_topic=data[driver_type]["robot_cmd_topic"],
            robot_status_topic=data[driver_type]["robot_status_topic"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "host": self.host,
            "pub_port": self.pub_port,
            "sub_port": self.sub_port,
            "robot_cmd_topic": self.robot_cmd_topic,
            "robot_status_topic": self.robot_status_topic,
        }


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
