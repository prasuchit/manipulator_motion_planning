import zmq


class ZmqSubscriber:
    def __init__(self, socket, topic):
        self.socket = socket
        self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)

    def recv_message(self):
        try:
            _ = self.socket.recv_string(flags=zmq.NOBLOCK)
            return self.socket.recv_json()
        except zmq.Again:
            # No new data received, don't latch
            return None
