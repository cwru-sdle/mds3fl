import socket, pickle, struct, threading, torch, time
from mds3fl.algorithm import fedavg, fedprox, feddyn_server_step

class RAWTCPServer:
    def __init__(self, host="127.0.0.1", port=5000, max_clients=5):
        self.host = host
        self.port = port
        self.max_clients = max_clients
        self.client_states = []
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(max_clients)
        print(f"[Server] Listening on {host}:{port}")

    def _recv_data(self, conn):
        raw_len = conn.recv(4)
        if not raw_len:
            return None
        msg_len = struct.unpack('>I', raw_len)[0]
        data = b''
        while len(data) < msg_len:
            packet = conn.recv(4096)
            if not packet:
                return None
            data += packet
        return pickle.loads(data)

    def _send_data(self, conn, obj):
        data = pickle.dumps(obj)
        conn.sendall(struct.pack('>I', len(data)) + data)

    def handle_client(self, conn, addr):
        print(f"[Server] Connected by {addr}")
        state_dict = self._recv_data(conn)
        if state_dict:
            self.client_states.append(state_dict)
            print(f"[Server] Received model from {addr}")

            # --- 聚合逻辑：简单平均 ---
            if len(self.client_states) > 0:
                avg_state = {}
                for k in self.client_states[0].keys():
                    avg_state[k] = torch.stack([c[k] for c in self.client_states]).mean(0)
            else:
                avg_state = state_dict

            # --- 发送更新模型 ---
            self._send_data(conn, avg_state)
            print(f"[Server] Sent aggregated model to {addr}")
        conn.close()

    def receive_all_clients(self, timeout=None):
        """兼容旧版接口：接收所有客户端模型（带可选超时）"""
        self.client_states.clear()
        threads = []
        self.server_socket.settimeout(timeout)
        print(f"[Server] Ready for up to {self.max_clients} clients.")
        for _ in range(self.max_clients):
            try:
                conn, addr = self.server_socket.accept()
                t = threading.Thread(target=self.handle_client, args=(conn, addr))
                t.start()
                threads.append(t)
            except socket.timeout:
                print("[Server] Timeout waiting for clients.")
                break
        for t in threads:
            t.join()
        return self.client_states
    
    def broadcast_model(self, model_state):
        """兼容旧版接口：向所有客户端广播聚合后的模型"""
        print(f"[Server] Broadcasting model to {self.max_clients} clients...")
        for _ in range(self.max_clients):
            try:
                conn, addr = self.server_socket.accept()
                self._send_data(conn, model_state)
                print(f"[Server] Sent global model to {addr}")
                conn.close()
            except Exception as e:
                print(f"[Server] Broadcast error: {e}")



class TCPServer:
    """Round-based Federated Learning TCP server wrapper.

    Fixes the old logic issues:
    - Do NOT aggregate and respond per-client.
    - Collect a full cohort (or until timeout), aggregate ONCE, then send the SAME model to all clients.
    - Avoid duplicating sockets/state; delegate networking to `TCPServer`.

    Protocol per round (single TCP connection per client per round):
      Client -> Server: send state_dict
      Server -> Client: send aggregated state_dict
      Server then closes the connection.
    """

    def __init__(self, model, algo="fedavg", alpha=0.1, networkstatus=False, host="127.0.0.1", port=5000, max_clients=5):
        self.model = model
        self.algo = algo.lower()
        self.alpha = alpha
        self.networkconnection = 1 if networkstatus else 0
        self.host = host
        self.port = port
        self.max_clients = max_clients

        # Create the underlying TCPServer only when networking is enabled.
        self.tcp_server = None
        if self.networkconnection == 1:
            self.tcp_server = RAWTCPServer(host=host, port=port, max_clients=max_clients)

        # FedDyn server-side state (only if algo == 'feddyn')
        if self.algo == "feddyn":
            self.w = {k: v.detach().clone() for k, v in model.state_dict().items()}
            self.g = {k: torch.zeros_like(v) for k, v in self.w.items()}

    def receive_all_clients(self, timeout=None):
        """Receive up to `max_clients` model states for a single round.

        Returns a tuple: (client_states, client_conns)
          - client_states: list[state_dict]
          - client_conns:  list[socket] (open connections to reply on)

        NOTE: This preserves the old name, but fixes the semantics.
        """
        if self.networkconnection != 1 or self.tcp_server is None:
            raise RuntimeError("TCP server is not enabled. Set networkstatus=True.")

        client_states = []
        client_conns = []

        # Accept loop (single thread is enough; each client sends one blob then waits for reply)
        self.tcp_server.server_socket.settimeout(timeout)
        print(f"[Server] Ready for up to {self.max_clients} clients (timeout={timeout}).")

        for _ in range(self.max_clients):
            try:
                conn, addr = self.tcp_server.server_socket.accept()
                print(f"[Server] Connected by {addr}")
                state_dict = self.tcp_server._recv_data(conn)
                if state_dict is None:
                    print(f"[Server] No data received from {addr}, closing.")
                    conn.close()
                    continue
                client_states.append(state_dict)
                client_conns.append(conn)
                print(f"[Server] Received model from {addr}")
            except socket.timeout:
                print("[Server] Timeout waiting for clients.")
                break

        return client_states, client_conns

    def run_round(self, timeout=None):
        """Run ONE federated round over TCP.

        - Wait for up to max_clients uploads (or until timeout)
        - Aggregate once
        - Reply the SAME aggregated model to all connected clients
        - Close all connections

        Returns the aggregated state_dict.
        """
        client_states, client_conns = self.receive_all_clients(timeout=timeout)

        if not client_states:
            print("[Server] No client updates collected this round.")
            for c in client_conns:
                try:
                    c.close()
                except Exception:
                    pass
            return None

        if self.algo == "fedavg":
            agg_state = fedavg(client_states)
        elif self.algo == "fedprox":
            # server-side FedProx is identical to FedAvg
            agg_state = fedavg(client_states)
        elif self.algo == "feddyn":
            agg_state, self.g = feddyn_server_step(
                client_states,
                self.w,
                self.g,
                self.alpha,
            )
            # update server-side global model
            self.w = {k: v.detach().clone() for k, v in agg_state.items()}
        else:
            raise NotImplementedError(f"Unsupported algorithm: {self.algo}")

        # Reply to all clients with the SAME aggregated model
        for conn in client_conns:
            try:
                self.tcp_server._send_data(conn, agg_state)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        print(f"[Server] Sent aggregated model to {len(client_conns)} clients.")
        return agg_state

    # Backward-compatible name; old code may call broadcast_model after receive_all_clients.
    # We keep it but make it a no-op helper that sends on provided conns.
    def broadcast_model(self, model_state, client_conns=None):
        """Send `model_state` to all provided connections and close them.

        Prefer `run_round()` which already performs collection+aggregation+reply.
        """
        if self.networkconnection != 1 or self.tcp_server is None:
            raise RuntimeError("TCP server is not enabled. Set networkstatus=True.")

        if client_conns is None:
            raise ValueError("client_conns must be provided; this server does not accept new connections for broadcast.")

        for conn in client_conns:
            try:
                self.tcp_server._send_data(conn, model_state)
            finally:
                try:
                    conn.close()
                except Exception:
                    pass

        print(f"[Server] Broadcasted model to {len(client_conns)} clients.")