from manipulator_motion_planning.motion_types import ControllerCommand, DriverCommand


class Controller:
    def __init__(self):
        self.ctrl_cmd = None

    def set_controller_command(self, ctrl_cmd: ControllerCommand):
        self.ctrl_cmd = ctrl_cmd

    def get_driver_cmd(self, driver_status, t):
        if self.ctrl_cmd is None:
            return None

        if t > self.ctrl_cmd.trajectory.get_duration():
            return None

        # Compute target position, and generate driver command
        return DriverCommand(
            target_joint_positions=self.ctrl_cmd.trajectory.get_position(t)
        )
