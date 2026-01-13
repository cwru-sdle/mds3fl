# import torch
# from federated_packet.network.tcp_server import TCPServer

# if __name__ == "__main__":
#     try:
#         print("[Server] Starting TCP server on 127.0.0.1:29500...")
#         # 初始化 TCP 服务器（这里假设 2 个客户端）
#         server = TCPServer(host="127.0.0.1", port=29500, max_clients=2)

#         print("\n[Server] Waiting for clients to send models...")
#         client_states = server.receive_all_clients()  # 等待所有客户端连接并接收数据
#         print(f"[Server] Received {len(client_states)} model(s).")

#         # 模拟聚合（简单平均）
#         avg_state = {}
#         for k in client_states[0].keys():
#             avg_state[k] = torch.stack([c[k] for c in client_states]).mean(0)

#         print("[Server] Aggregation done, broadcasting model...")
#         server.broadcast_model(avg_state)
#         print("[Server] Broadcast complete.")

#     except KeyboardInterrupt:
#         print("\n[Server] KeyboardInterrupt received. Shutting down server gracefully.")
import torch
from federated_packet.network.tcp_server import TCPServer

if __name__ == "__main__":
    try:
        print("[Server] Starting TCP server on 127.0.0.1:29500...")
        server = TCPServer(host="127.0.0.1", port=29500, max_clients=2)

        print("\n[Server] Waiting for clients to send models...")
        client_states = server.receive_all_clients()
        print(f"[Server] Received {len(client_states)} model(s).")

        avg_state = {}
        for k in client_states[0].keys():
            avg_state[k] = torch.stack([c[k] for c in client_states]).mean(0)

        print("[Server] Aggregation done. Round complete.")
    except KeyboardInterrupt:
        print("\n[Server] Shutdown gracefully.")