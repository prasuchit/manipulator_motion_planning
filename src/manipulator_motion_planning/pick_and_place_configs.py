from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np
import yaml
from numpy.typing import NDArray


class PickAndPlaceActionType(Enum):
    NONE = auto()
    READY = auto()
    MOVE_TO_PRE_PICK = auto()
    MOVE_TO_PICK = auto()
    PICK_ITEM = auto()
    MOVE_TO_POST_PICK = auto()
    MOVE_TO_PRE_PLACE = auto()
    MOVE_TO_PLACE = auto()
    PLACE_ITEM = auto()
    MOVE_TO_POST_PLACE = auto()
    MOVE_TO_HOME = auto()


class ActionState(Enum):
    RUNNING = auto()
    IDLE = auto()


@dataclass
class JointStateConfig:
    home_pose_js: NDArray


@dataclass
class TaskStateConfig:
    # Pick
    pre_pick_pose_ts: NDArray
    pick_pose_ts: NDArray
    post_pick_pose_ts: NDArray

    # Place
    pre_place_pose_ts: NDArray
    place_pose_ts: NDArray
    post_place_pose_ts: NDArray


@dataclass
class ActionDurationConfig:
    move_to_pre_pick_duration_s: float
    move_to_pre_place_duration_s: float
    move_to_home_duration_s: float
    move_to_pick_duration_s: float
    pick_duration_s: float
    move_to_post_pick_duration_s: float
    move_to_place_duration_s: float
    place_duration_s: float
    move_to_post_place_duration_s: float


@dataclass
class PickAndPlaceActionConfig:
    joint_states: JointStateConfig
    task_states: TaskStateConfig
    action_durations: ActionDurationConfig

    @staticmethod
    def load_from_yaml(yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        joint = data["joint_states"]
        task = data["task_states"]
        durations = data["action_durations"]

        return PickAndPlaceActionConfig(
            joint_states=JointStateConfig(home_pose_js=np.array(joint["home_pose_js"])),
            task_states=TaskStateConfig(
                pre_pick_pose_ts=np.array(task["pre_pick_pose_ts"]),
                pick_pose_ts=np.array(task["pick_pose_ts"]),
                post_pick_pose_ts=np.array(task["post_pick_pose_ts"]),
                pre_place_pose_ts=np.array(task["pre_place_pose_ts"]),
                place_pose_ts=np.array(task["place_pose_ts"]),
                post_place_pose_ts=np.array(task["post_place_pose_ts"]),
            ),
            action_durations=ActionDurationConfig(
                move_to_pre_pick_duration_s=durations["move_to_pre_pick_duration_s"],
                move_to_pre_place_duration_s=durations["move_to_pre_place_duration_s"],
                move_to_home_duration_s=durations["move_to_home_duration_s"],
                move_to_pick_duration_s=durations["move_to_pick_duration_s"],
                pick_duration_s=durations["pick_duration_s"],
                move_to_post_pick_duration_s=durations["move_to_post_pick_duration_s"],
                move_to_place_duration_s=durations["move_to_place_duration_s"],
                place_duration_s=durations["place_duration_s"],
                move_to_post_place_duration_s=durations[
                    "move_to_post_place_duration_s"
                ],
            ),
        )


class GripperState(Enum):
    OPEN = 0
    CLOSED = 255


@dataclass
class Stage:
    """Represents one stage of a pick-and-place sequence."""

    name: str
    target_pos: Optional[NDArray] = None  # Task-space goal (3D)
    target_rot_func: Optional[Callable[[], NDArray]] = (
        None  # Function returning rotation matrix
    )
    duration_s: float = 1.0
    gripper_state: GripperState = GripperState.OPEN
