import numpy as np

from manipulator_motion_planning.controller import Controller
from manipulator_motion_planning.motion_planning.trajectory_generator import CubicTrajectory
from manipulator_motion_planning.motion_types import ControllerCommand
from manipulator_motion_planning.mujoco_model_manager import MujocoModelManager
from manipulator_motion_planning.pick_and_place import PickAndPlaceActionConfig, PickAndPlaceActionPlanner
from manipulator_motion_planning.simulation_driver import SimulationDriver


def generate_controller_command(
    current_joint_positions, target_joint_positions, duration_s
):
    trajectory = CubicTrajectory(
        start_pos=current_joint_positions,
        end_pos=target_joint_positions,
        start_vel=np.zeros(len(current_joint_positions)),
        end_vel=np.zeros(len(current_joint_positions)),
        duration=duration_s,
    )

    return ControllerCommand(trajectory=trajectory)


def main():
    # Create a sim_driver so we can interact with the simulator
    sim_driver = SimulationDriver()
    sim_driver.wait_for_initialization()

    # Create our action
    pick_n_place_cfg = PickAndPlaceActionConfig.load_from_yaml(
        "configs/pick_and_place_config.yaml"
    )
    mm = MujocoModelManager("models/main.xml")
    pick_n_place_planner = PickAndPlaceActionPlanner(pick_n_place_cfg, mm)

    # Create a controller
    controller = Controller()

    # Core controller loop
    loop_rate = 2 * 1e-3

    loop_start = sim_driver.now()
    action_start = None
    while not pick_n_place_planner.done():
        cycle_start = sim_driver.now()
        t = cycle_start - loop_start
        if action_start is None:
            action_start = t

        # Get current driver command
        driver_status = sim_driver.get_status()
        driver_cmd = controller.get_driver_cmd(driver_status, t - action_start)

        # Need to clean up this handling a bit
        if driver_cmd is None:
            # For now we will proxy this as action complete
            action_start = None  # This will automatically reset our action start
            pick_n_place_planner.tick()
            controller.set_controller_command(pick_n_place_planner.controller_command)
        else:
            # Send driver cmd
            sim_driver.send_command(driver_cmd)

        # Wait until next loop
        while sim_driver.now() - cycle_start < loop_rate:
            continue

    print("Pick and place done!!")


if __name__ == "__main__":
    main()
