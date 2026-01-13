import torch

def fedavg(client_states):
    """FedAvg aggregation"""
    new_state = {}
    for k in client_states[0].keys():
        new_state[k] = torch.stack([c[k] for c in client_states]).mean(0)
    return new_state

def fedadam(client_states, global_model_state, m_t, v_t, beta1=0.9, beta2=0.999, lr=0.001, epsilon=1e-8):
    """FedAdam aggregation"""
    new_state = {}
    g_t = {}
    for k in client_states[0].keys():
        g_t[k] = torch.stack([c[k] for c in client_states]).mean(0) - global_model_state[k]
        m_t[k] = beta1 * m_t[k] + (1 - beta1) * g_t[k]
        v_t[k] = beta2 * v_t[k] + (1 - beta2) * (g_t[k] ** 2)
        m_hat = m_t[k] / (1 - beta1)
        v_hat = v_t[k] / (1 - beta2)
        new_state[k] = global_model_state[k] + lr * m_hat / (torch.sqrt(v_hat) + epsilon)
    return new_state, m_t, v_t

def fedprox(client_states, global_model_state, mu=0.01):
    """FedProx: penalizes local divergence from global model"""
    new_state = {}
    for k in client_states[0].keys():
        avg = torch.stack([c[k] for c in client_states]).mean(0)
        prox = global_model_state[k]
        new_state[k] = avg - mu * (avg - prox)
    return new_state

def feddyn_server_step(client_states, w_t, g_t, alpha):
    """FedDyn server-side aggregation step.

    Args:
        client_states: list of client state_dicts (w_i)
        w_t: dict, current global model at server
        g_t: dict, current dynamic correction term
        alpha: float, FedDyn hyperparameter

    Returns:
        new_w: dict, updated global model w_{t+1}
        new_g: dict, updated correction term g_{t+1}
    """
    # w_bar = average(w_i)
    w_bar = {}
    for k in client_states[0].keys():
        w_bar[k] = torch.stack([c[k] for c in client_states]).mean(0)

    # g_{t+1} = g_t + alpha * (w_bar - w_t)
    new_g = {}
    for k in w_bar.keys():
        new_g[k] = g_t[k] + alpha * (w_bar[k] - w_t[k])

    # w_{t+1} = w_bar - (1/alpha) * g_{t+1}
    new_w = {}
    for k in w_bar.keys():
        new_w[k] = w_bar[k] - (1.0 / alpha) * new_g[k]

    return new_w, new_g
