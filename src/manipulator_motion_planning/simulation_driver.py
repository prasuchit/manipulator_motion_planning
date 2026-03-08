import time

from manipulator_motion_planning.motion_types import DriverCommand, DriverStatus
from manipulator_motion_planning.zmq_common.publisher import ZmqPublisher
from manipulator_motion_planning.zmq_common.subscriber import ZmqSubscriber
from manipulator_motion_planning.zmq_common.utils import get_pub_socket, get_sub_socket, to_zmq_msg

HOST = "localhost"
PUB_PORT = 5555
SUB_PORT = 5556
ROBOT_CMD_TOPIC = "robot_cmd"
ROBOT_STATUS_TOPIC = "robot_status"


class SimulationDriver:
    def __init__(self):
        pub_socket = get_pub_socket(HOST, PUB_PORT)
        sub_socket = get_sub_socket(HOST, SUB_PORT)

        # Setup command publisher
        self.robot_cmd_pub = ZmqPublisher(pub_socket, ROBOT_CMD_TOPIC)
        self.robot_status_sub = ZmqSubscriber(sub_socket, ROBOT_STATUS_TOPIC)

    def wait_for_initialization(self):
        print("Waiting for valid status")
        while self.get_status() is None:
            time.sleep(1)
            continue
        print("Got valid status!")

    def send_command(self, cmd: DriverCommand):
        cmd_msg = to_zmq_msg(cmd.to_dict())
        self.robot_cmd_pub.send_message(cmd_msg)

    def get_current_joint_positions(self):
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
