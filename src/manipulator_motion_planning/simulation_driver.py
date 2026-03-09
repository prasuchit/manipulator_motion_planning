import time

from numpy.typing import NDArray

from manipulator_motion_planning.motion_types import (
    DriverCommand,
    DriverConfig,
    DriverStatus,
)
from manipulator_motion_planning.zmq_common.publisher import ZmqPublisher
from manipulator_motion_planning.zmq_common.subscriber import ZmqSubscriber
from manipulator_motion_planning.zmq_common.utils import (
    get_pub_socket,
    get_sub_socket,
    to_zmq_msg,
)


class SimulationDriver:
    def __init__(self, cfg: DriverConfig):
        self.cfg = cfg
        pub_socket = get_pub_socket(self.cfg.host, self.cfg.pub_port)
        sub_socket = get_sub_socket(self.cfg.host, self.cfg.sub_port)

        # Setup command publisher
        self.robot_cmd_pub = ZmqPublisher(pub_socket, self.cfg.robot_cmd_topic)
        self.robot_status_sub = ZmqSubscriber(sub_socket, self.cfg.robot_status_topic)

    def wait_for_initialization(self) -> None:
        print("Waiting for valid status")
        while self.get_status() is None:
            time.sleep(1)
            continue
        print("Got valid status!")

    def send_command(self, cmd: DriverCommand) -> None:
        cmd_msg = to_zmq_msg(cmd.to_dict())
        self.robot_cmd_pub.send_message(cmd_msg)

    def get_current_joint_positions(self) -> NDArray:
        driver_status = self.get_status()
        if driver_status is None:
            return None
        return driver_status.current_joint_positions

    def get_status(self) -> DriverStatus:
        status_json = self.robot_status_sub.recv_message()
        if status_json is None:
            return None

        return DriverStatus.from_dict(status_json)

    def now(self) -> float:
        max_timeout_s = 1  #
        loop_start_s = time.perf_counter()
        while True:
            if time.perf_counter() - loop_start_s > max_timeout_s:
                print("Max timeout while waiting for status")
                return None

            status_json = self.robot_status_sub.recv_message()
            if status_json is not None:
                break
            time.sleep(0.001)

        return status_json["time_s"]
