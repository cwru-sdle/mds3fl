import socket, pickle, struct

# class TCPClient:
#     def __init__(self, server_host="127.0.0.1", server_port=5000):
#         self.host = server_host
#         self.port = server_port

#     def _send_data(self, sock, obj):
#         data = pickle.dumps(obj)
#         sock.sendall(struct.pack('>I', len(data)) + data)

#     def _recv_data(self, sock):
#         raw_len = sock.recv(4)
#         if not raw_len:
#             return None
#         msg_len = struct.unpack('>I', raw_len)[0]
#         data = b''
#         while len(data) < msg_len:
#             packet = sock.recv(4096)
#             if not packet:
#                 return None
#             data += packet
#         return pickle.loads(data)

#     def send_model_and_get_update(self, model_state):
#         """Send local model to server and receive updated model in one connection"""
#         with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
#             s.connect((self.host, self.port))
#             self._send_data(s, model_state)
#             # Immediately wait for the response from the server
#             updated_state = self._recv_data(s)
#         return updated_state
    


#         pass

class TCPClient:
    """Minimal TCP client for custom federated-learning packets.

    Packet protocol (same for fedavg / fedprox / feddyn):
      Client -> Server: local model state_dict
      Server -> Client: aggregated model state_dict

    This client is intentionally algorithm-agnostic at the network layer.
    """

    def __init__(self, server_host="127.0.0.1", server_port=5000, algo="fedavg"):
        self.host = server_host
        self.port = server_port
        self.algo = algo.lower()

        if self.algo not in ("fedavg", "fedprox", "feddyn"):
            raise ValueError(f"Unsupported algorithm: {self.algo}")

    def _send_data(self, sock, obj):
        data = pickle.dumps(obj)
        sock.sendall(struct.pack('>I', len(data)) + data)

    def _recv_data(self, sock):
        raw_len = sock.recv(4)
        if not raw_len:
            return None
        msg_len = struct.unpack('>I', raw_len)[0]
        data = b''
        while len(data) < msg_len:
            packet = sock.recv(4096)
            if not packet:
                return None
            data += packet
        return pickle.loads(data)

    def run_round(self, local_model_state):
        """Run ONE federated round from the client side.

        Args:
            local_model_state: state_dict of the local model

        Returns:
            updated_state: state_dict returned by the server
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))

            # Send local model packet
            self._send_data(sock, local_model_state)

            # Wait for aggregated model packet
            updated_state = self._recv_data(sock)

        return updated_state