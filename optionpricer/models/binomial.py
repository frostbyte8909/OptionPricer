import numpy as np

def build_tree(S, K, T, r, sigma, N=100, option_type="call", american=False):
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u
    p  = (np.exp(r * dt) - d) / (u - d)
    df = np.exp(-r * dt)

    S_T    = S * u**np.arange(N + 1) * d**(N - np.arange(N + 1))
    option = np.maximum(S_T - K, 0) if option_type == "call" else np.maximum(K - S_T, 0)

    for i in range(N - 1, -1, -1):
        option[:-1] = df * (p * option[1:] + (1 - p) * option[:-1])
        if american:
            j = np.arange(i + 1)
            S_i = S * u**j * d**(i - j)
            intrinsic = np.maximum(K - S_i, 0) if option_type == "put" else np.maximum(S_i - K, 0)
            option[:i+1] = np.maximum(option[:i+1], intrinsic)
    return option[0]
