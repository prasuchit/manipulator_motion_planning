import zmq


class ZmqPublisher:
    def __init__(self, socket, topic):
        self.socket = socket
        self.topic = topic

    def send_message(self, msg):
        self.socket.send_string(self.topic, zmq.SNDMORE)
        self.socket.send_json(msg)
