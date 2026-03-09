import time

import cv2
import mujoco
import mujoco.viewer

from manipulator_motion_planning import PROJECT_ROOT
from manipulator_motion_planning.model_manager.mujoco_model_manager import (
    MujocoModelManager,
)
from manipulator_motion_planning.motion_types import DriverConfig
from manipulator_motion_planning.zmq_common.publisher import ZmqPublisher
from manipulator_motion_planning.zmq_common.subscriber import ZmqSubscriber
from manipulator_motion_planning.zmq_common.utils import get_pub_socket, get_sub_socket

MS_TO_S = 1e-3
RECORD = True
VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 960

def main():
    sim_driver_config = DriverConfig.load_from_yaml(
        f"{PROJECT_ROOT}/configs/simulation_driver_config.yaml", driver_type="client"
    )
    pub_socket = get_pub_socket(sim_driver_config.host, sim_driver_config.pub_port)
    sub_socket = get_sub_socket(sim_driver_config.host, sim_driver_config.sub_port)
    robot_cmd_sub = ZmqSubscriber(sub_socket, sim_driver_config.robot_cmd_topic)
    robot_status_pub = ZmqPublisher(pub_socket, sim_driver_config.robot_status_topic)

    mm = MujocoModelManager(scene_path=f"{PROJECT_ROOT}/models/main.xml")
    home = [1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0]
    mm.data.qpos[:6] = home
    mm.data.ctrl[:6] = home
    mujoco.mj_forward(mm.model, mm.data)

    status_publish_interval_s = 2 * MS_TO_S

    if RECORD:
        renderer = mujoco.Renderer(mm.model, height=VIDEO_HEIGHT, width=VIDEO_WIDTH)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        video = cv2.VideoWriter(
            "recording.avi", fourcc, VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT)
        )
        frame_interval = 1.0 / VIDEO_FPS
        last_frame_time = 0.0

    try:
        with mujoco.viewer.launch_passive(
            mm.model, mm.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            last_pub_time = mm.data.time

            while viewer.is_running():
                step_start = time.perf_counter()

                cmd = robot_cmd_sub.recv_message()
                if cmd is not None:
                    cmd = cmd["value"]
                    mm.data.ctrl[:7] = cmd["target_joint_positions"]

                mujoco.mj_step(mm.model, mm.data)

                now = mm.data.time
                if now - last_pub_time >= status_publish_interval_s:
                    robot_status = {
                        "time_s": mm.data.time,
                        "current_joint_positions": mm.get_joint_positions(),
                        "current_joint_velocities": mm.get_joint_velocities(),
                        "current_joint_accelerations": mm.get_joint_accelerations(),
                    }
                    robot_status_pub.send_message(robot_status)
                    last_pub_time = now

                if RECORD and (now - last_frame_time >= frame_interval):
                    renderer.update_scene(mm.data, camera=viewer.cam)
                    frame = renderer.render()
                    video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    last_frame_time = now

                viewer.sync()

                elapsed = time.perf_counter() - step_start
                remaining = mm.model.opt.timestep - elapsed
                if remaining > 0:
                    time.sleep(remaining)

    finally:
        if RECORD:
            video.release()
            renderer.close()
            print("Saved recording.avi")


if __name__ == "__main__":
    main()
