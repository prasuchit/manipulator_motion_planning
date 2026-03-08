from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import yaml
from numpy.typing import NDArray

from manipulator_motion_planning.controller import ControllerCommand
from manipulator_motion_planning.motion_planning.trajectory_generator import CubicTrajectory


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
class PickAndPlaceActionConfig:
    home_pose_js: NDArray
    pre_pick_pose_ts: NDArray
    pick_pose_ts: NDArray
    post_pick_pose_ts: NDArray
    pre_place_pose_ts: NDArray
    place_pose_ts: NDArray
    post_place_pose_ts: NDArray

    move_to_pre_pick_duration_s: float
    move_to_pick_duration_s: float
    pick_duration_s: float
    move_to_post_pick_duration_s: float
    move_to_pre_place_duration_s: float
    move_to_place_duration_s: float
    place_duration_s: float
    move_to_post_place_duration_s: float
    move_to_home_duration_s: float

    @staticmethod
    def load_from_yaml(yaml_file):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)

        return PickAndPlaceActionConfig(
            home_pose_js=np.array(data["home_pose_js"]),
            pre_pick_pose_ts=np.array(data["pre_pick_pose_ts"]),
            pick_pose_ts=np.array(data["pick_pose_ts"]),
            post_pick_pose_ts=np.array(data["post_pick_pose_ts"]),
            pre_place_pose_ts=np.array(data["pre_place_pose_ts"]),
            place_pose_ts=np.array(data["place_pose_ts"]),
            post_place_pose_ts=np.array(data["post_place_pose_ts"]),
            move_to_pre_pick_duration_s=data["move_to_pre_pick_duration_s"],
            move_to_pick_duration_s=data["move_to_pick_duration_s"],
            pick_duration_s=data["pick_duration_s"],
            move_to_post_pick_duration_s=data["move_to_post_pick_duration_s"],
            move_to_pre_place_duration_s=data["move_to_pre_place_duration_s"],
            move_to_place_duration_s=data["move_to_place_duration_s"],
            place_duration_s=data["place_duration_s"],
            move_to_post_place_duration_s=data["move_to_post_place_duration_s"],
            move_to_home_duration_s=data["move_to_home_duration_s"],
        )


class PickAndPlaceActionPlanner:
    def __init__(self, cfg, model_manager):
        self.cfg = cfg
        self.next_action = PickAndPlaceActionType.READY
        self.model_manager = model_manager
        self.controller_command = None
        self._done = False

    def _print_current_action(self):
        print(f"Executing {self.next_action.name} action...")

    def tick(self):
        if self.next_action == PickAndPlaceActionType.READY:
            # Transition immediately to pre_pick
            self.next_action = PickAndPlaceActionType.MOVE_TO_PRE_PICK
        elif self.next_action == PickAndPlaceActionType.MOVE_TO_PRE_PICK:
            self._move_to_pre_pick()
        elif self.next_action == PickAndPlaceActionType.MOVE_TO_PICK:
            self._move_to_pick()
        elif self.next_action == PickAndPlaceActionType.PICK_ITEM:
            self._pick_item()
        elif self.next_action == PickAndPlaceActionType.MOVE_TO_POST_PICK:
            self._move_to_post_pick()
        elif self.next_action == PickAndPlaceActionType.MOVE_TO_PRE_PLACE:
            self._move_to_pre_place()
        elif self.next_action == PickAndPlaceActionType.MOVE_TO_PLACE:
            self._move_to_place()
        elif self.next_action == PickAndPlaceActionType.PLACE_ITEM:
            self._place_item()
        elif self.next_action == PickAndPlaceActionType.MOVE_TO_POST_PLACE:
            self._move_to_post_place()
        elif self.next_action == PickAndPlaceActionType.MOVE_TO_HOME:
            self._move_to_home()
        elif self.next_action == PickAndPlaceActionType.NONE:
            self._handle_none_action()

    def _rot_from_js_pose(self, js_pose):
        # Do FK
        T = self.model_manager.fk(js_pose)
        return T[:3, :3]

    def _move_to_pre_pick(self):
        self._print_current_action()

        # Compute pre_pick_pose_js
        self.pre_pick_pose_js = self.model_manager.ik(
            target_pos=self.cfg.pre_pick_pose_ts,
            target_rot=self._rot_from_js_pose(self.cfg.home_pose_js),
            qinit=self.cfg.home_pose_js,
        )
        # Keep gripper open
        self.pre_pick_pose_js = np.append(self.pre_pick_pose_js, 0)

        self._generate_command(
            start_joint_positions=self.cfg.home_pose_js,
            end_joint_positions=self.pre_pick_pose_js,
            traj_duration_s=self.cfg.move_to_pre_pick_duration_s,
        )

        self.next_action = PickAndPlaceActionType.MOVE_TO_PICK

    def _move_to_pick(self):
        self._print_current_action()

        # Compute pick_pose_js
        self.pick_pose_js = self.model_manager.ik(
            target_pos=self.cfg.pick_pose_ts,
            target_rot=self._rot_from_js_pose(self.pre_pick_pose_js),
            qinit=self.pre_pick_pose_js,
        )
        # Keep gripper open
        self.pick_pose_js = np.append(self.pick_pose_js, 0)

        self._generate_command(
            start_joint_positions=self.pre_pick_pose_js,
            end_joint_positions=self.pick_pose_js,
            traj_duration_s=self.cfg.move_to_pick_duration_s,
        )

        self.next_action = PickAndPlaceActionType.PICK_ITEM

    def _pick_item(self):
        self._print_current_action()

        self.picking_pose_js = self.pick_pose_js
        self.picking_pose_js[-1] = 255  # Close all the way

        self._generate_command(
            start_joint_positions=self.pick_pose_js,
            end_joint_positions=self.picking_pose_js,
            traj_duration_s=self.cfg.pick_duration_s,
        )

        self.next_action = PickAndPlaceActionType.MOVE_TO_POST_PICK

    def _move_to_post_pick(self):
        self._print_current_action()

        # Compute post pick pose js
        self.post_pick_pose_js = self.model_manager.ik(
            target_pos=self.cfg.post_pick_pose_ts,
            target_rot=self._rot_from_js_pose(self.picking_pose_js),
            qinit=self.picking_pose_js,
        )
        # Keep gripper closed
        self.post_pick_pose_js = np.append(self.post_pick_pose_js, 255)

        self._generate_command(
            start_joint_positions=self.picking_pose_js,
            end_joint_positions=self.post_pick_pose_js,
            traj_duration_s=self.cfg.move_to_post_pick_duration_s,
        )

        self.next_action = PickAndPlaceActionType.MOVE_TO_PRE_PLACE

    def _move_to_pre_place(self):
        self._print_current_action()

        # Compute pre place pose js
        self.pre_place_pose_js = self.model_manager.ik(
            target_pos=self.cfg.pre_place_pose_ts,
            target_rot=self._rot_from_js_pose(self.post_pick_pose_js),
            qinit=self.post_pick_pose_js,
        )
        # Keep gripper closed
        self.pre_place_pose_js = np.append(self.pre_place_pose_js, 255)

        self._generate_command(
            start_joint_positions=self.post_pick_pose_js,
            end_joint_positions=self.pre_place_pose_js,
            traj_duration_s=self.cfg.move_to_pre_place_duration_s,
        )

        self.next_action = PickAndPlaceActionType.MOVE_TO_PLACE

    def _move_to_place(self):
        self._print_current_action()

        # Compute place pose js
        self.place_pose_js = self.model_manager.ik(
            target_pos=self.cfg.place_pose_ts,
            target_rot=self._rot_from_js_pose(self.pre_place_pose_js),
            qinit=self.pre_place_pose_js,
        )
        # Keep gripper closed
        self.place_pose_js = np.append(self.place_pose_js, 255)

        self._generate_command(
            start_joint_positions=self.pre_place_pose_js,
            end_joint_positions=self.place_pose_js,
            traj_duration_s=self.cfg.move_to_place_duration_s,
        )

        self.next_action = PickAndPlaceActionType.PLACE_ITEM

    def _place_item(self):
        self._print_current_action()

        # Compute place pose
        self.placing_pose_js = self.place_pose_js
        self.placing_pose_js[-1] = 0  # Open it now fully
        self._generate_command(
            start_joint_positions=self.place_pose_js,
            end_joint_positions=self.placing_pose_js,
            traj_duration_s=self.cfg.place_duration_s,
        )

        self.next_action = PickAndPlaceActionType.MOVE_TO_POST_PLACE

    def _move_to_post_place(self):
        self._print_current_action()

        # Compute post place
        self.post_place_pose_js = self.model_manager.ik(
            target_pos=self.cfg.post_place_pose_ts,
            target_rot=self._rot_from_js_pose(self.placing_pose_js),
            qinit=self.placing_pose_js,
        )
        # Keep the gripper open
        self.post_place_pose_js = np.append(self.post_place_pose_js, 0)

        self._generate_command(
            start_joint_positions=self.placing_pose_js,
            end_joint_positions=self.post_place_pose_js,
            traj_duration_s=self.cfg.move_to_post_place_duration_s,
        )

        self.next_action = PickAndPlaceActionType.MOVE_TO_HOME

    def _move_to_home(self):
        self._print_current_action()

        self._generate_command(
            start_joint_positions=self.post_place_pose_js,
            end_joint_positions=self.cfg.home_pose_js,
            traj_duration_s=self.cfg.move_to_home_duration_s,
        )

        self.next_action = PickAndPlaceActionType.NONE

    def _handle_none_action(self):
        self._print_current_action()
        self.controller_command = None
        self._done = True

    def _generate_command(
        self, start_joint_positions, end_joint_positions, traj_duration_s
    ):
        num_dofs = len(start_joint_positions)
        zeros = np.zeros(num_dofs)
        trajectory = CubicTrajectory(
            start_pos=start_joint_positions,
            end_pos=end_joint_positions,
            start_vel=zeros,
            end_vel=zeros,
            duration=traj_duration_s,
        )
        self.controller_command = ControllerCommand(trajectory=trajectory)

    def done(self):
        return self._done
