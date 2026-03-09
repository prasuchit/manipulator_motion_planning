from typing import Any, Dict, List

import numpy as np

from manipulator_motion_planning.controller import ControllerCommand
from manipulator_motion_planning.model_manager.base import RobotModelManagerBase
from manipulator_motion_planning.motion_planning.trajectory_generator import (
    CubicTrajectory,
)
from manipulator_motion_planning.pick_and_place_configs import GripperState, Stage


def compute_rotation_from_js(
    model_manager: RobotModelManagerBase, js: np.ndarray
) -> np.ndarray:
    """
    Helper to compute the end-effector rotation matrix from a joint-space pose.
    """
    T = model_manager.fk(js)
    return T[:3, :3]  # Extract rotation part


def create_pick_and_place_stages(
    cfg: Dict, model_manager: RobotModelManagerBase
) -> List[Stage]:
    """
    Returns an ordered list of stages for the entire pick-and-place operation.
    """
    home_js = cfg.joint_states.home_pose_js

    return [
        Stage(
            name="MOVE_TO_PRE_PICK",
            target_pos=cfg.task_states.pre_pick_pose_ts,
            target_rot_func=lambda: compute_rotation_from_js(model_manager, home_js),
            duration_s=cfg.action_durations.move_to_pre_pick_duration_s,
            gripper_state=GripperState.OPEN,
        ),
        Stage(
            name="MOVE_TO_PICK",
            target_pos=cfg.task_states.pick_pose_ts,
            target_rot_func=lambda: compute_rotation_from_js(model_manager, home_js),
            duration_s=cfg.action_durations.move_to_pick_duration_s,
            gripper_state=GripperState.OPEN,
        ),
        Stage(
            name="PICK_ITEM",
            target_pos=cfg.task_states.pick_pose_ts,
            target_rot_func=None,
            duration_s=cfg.action_durations.pick_duration_s,
            gripper_state=GripperState.CLOSED,
        ),
        Stage(
            name="MOVE_TO_POST_PICK",
            target_pos=cfg.task_states.post_pick_pose_ts,
            target_rot_func=lambda: compute_rotation_from_js(model_manager, home_js),
            duration_s=cfg.action_durations.move_to_post_pick_duration_s,
            gripper_state=GripperState.CLOSED,
        ),
        Stage(
            name="MOVE_TO_PRE_PLACE",
            target_pos=cfg.task_states.pre_place_pose_ts,
            target_rot_func=lambda: compute_rotation_from_js(model_manager, home_js),
            duration_s=cfg.action_durations.move_to_pre_place_duration_s,
            gripper_state=GripperState.CLOSED,
        ),
        Stage(
            name="MOVE_TO_PLACE",
            target_pos=cfg.task_states.place_pose_ts,
            target_rot_func=lambda: compute_rotation_from_js(model_manager, home_js),
            duration_s=cfg.action_durations.move_to_place_duration_s,
            gripper_state=GripperState.CLOSED,
        ),
        Stage(
            name="PLACE_ITEM",
            target_pos=cfg.task_states.place_pose_ts,
            target_rot_func=None,
            duration_s=cfg.action_durations.place_duration_s,
            gripper_state=GripperState.OPEN,
        ),
        Stage(
            name="MOVE_TO_POST_PLACE",
            target_pos=cfg.task_states.post_place_pose_ts,
            target_rot_func=lambda: compute_rotation_from_js(model_manager, home_js),
            duration_s=cfg.action_durations.move_to_post_place_duration_s,
            gripper_state=GripperState.OPEN,
        ),
        Stage(
            name="MOVE_TO_HOME",
            target_pos=None,  # Home in joint space
            target_rot_func=None,
            duration_s=cfg.action_durations.move_to_home_duration_s,
            gripper_state=GripperState.OPEN,
        ),
    ]


class PickAndPlacePlanner:
    def __init__(self, cfg: Dict[str, Any], model_manager: RobotModelManagerBase):
        self.cfg = cfg
        self.model_manager = model_manager
        self.stages = create_pick_and_place_stages(cfg, model_manager)
        self.current_stage_idx = 0
        self.controller_command = None
        
    def ensure_gripper(self, q: np.ndarray, gripper_value: int) -> np.ndarray:
        """Append gripper value if not already present."""
        if len(q) == self.model_manager.n_dof:
            return np.append(q, gripper_value)
        return q
    
    def tick(self):
        """Compute the controller command for the current stage."""
        if self.current_stage_idx >= len(self.stages):
            self.controller_command = None
            return

        stage = self.stages[self.current_stage_idx]

        # Determine joint-space target
        if stage.target_pos is not None:
            qinit = (
                self.cfg.joint_states.home_pose_js
                if self.current_stage_idx == 0
                else self.q_current
            )
            target_rot = stage.target_rot_func() if stage.target_rot_func else None
            q_target = self.model_manager.ik(
                target_pos=stage.target_pos, target_rot=target_rot, qinit=qinit
            )
            # Safe append of gripper
            q_target = self.ensure_gripper(q_target, stage.gripper_state.value)
        else:
            # If no task-space target, go to home joint positions
            q_target = self.cfg.joint_states.home_pose_js.copy()

        # Generate trajectory
        self.controller_command = self._generate_trajectory(
            start_joint_positions=getattr(
                self, "q_current", self.cfg.joint_states.home_pose_js
            ),
            end_joint_positions=q_target,
            duration_s=stage.duration_s,
        )

        # Advance current stage
        self.q_current = q_target.copy()
        self.current_stage_idx += 1

    def _generate_trajectory(
        self, start_joint_positions, end_joint_positions, duration_s
    ):
        """Return a CubicTrajectory wrapped in ControllerCommand."""
        traj = CubicTrajectory(
            start_pos=start_joint_positions,
            end_pos=end_joint_positions,
            start_vel=np.zeros(len(start_joint_positions)),
            end_vel=np.zeros(len(end_joint_positions)),
            duration=duration_s,
        )
        return ControllerCommand(trajectory=traj)

    def done(self):
        return self.current_stage_idx >= len(self.stages)
