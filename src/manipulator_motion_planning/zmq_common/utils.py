import time
from typing import Any

import zmq


def to_zmq_msg(msg: Any):
    return {"timestamp": time.time(), "value": msg}


def get_pub_socket(host, port):
    ctx = zmq.Context()
    pub_socket = ctx.socket(zmq.PUB)
    pub_socket.bind(f"tcp://{host}:{port}")
    return pub_socket


def get_sub_socket(host, port):
    ctx = zmq.Context()
    sub_socket = ctx.socket(zmq.SUB)
    sub_socket.connect(f"tcp://{host}:{port}")
    return sub_socket
