import numpy as np
from numba import njit
from typing import Union

try:
    from optionpricer.models._binomial_cy import _build_tree_cython as _build_tree_core
except ImportError:
    @njit(cache=True, fastmath=True)
    def _build_tree_core(S, K, u, d, df, p, N, is_call, american):
        option = np.empty(N + 1)
        for j in range(N + 1):
            S_ij = S * (u ** (N - j)) * (d ** j)
            if is_call:
                option[j] = S_ij - K if S_ij > K else 0.0
            else:
                option[j] = K - S_ij if K > S_ij else 0.0
                
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                option[j] = df * (p * option[j + 1] + (1 - p) * option[j])
                if american:
                    S_ij = S * (u ** (i - j)) * (d ** j)
                    if is_call:
                        intrinsic = S_ij - K if S_ij > K else 0.0
                    else:
                        intrinsic = K - S_ij if K > S_ij else 0.0
                    if intrinsic > option[j]:
                        option[j] = intrinsic
        return option[0]

def _build_tree_scalar(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, N: int = 1000, option_type: str = "call", american: bool = False) -> float:
    is_call = option_type == "call"
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    df = np.exp(-r * dt)
    p  = (np.exp((r - q) * dt) - d) / (u - d)

    return _build_tree_core(float(S), float(K), float(u), float(d), float(df), float(p), int(N), is_call, american)

def build_tree(S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], r: Union[float, np.ndarray], sigma: Union[float, np.ndarray], q: Union[float, np.ndarray] = 0.0, N: int = 1000, option_type: str = "call", american: bool = False) -> Union[float, np.ndarray]:
    if any(isinstance(x, np.ndarray) for x in (S, K, T, r, sigma, q)):
        v_tree = np.vectorize(_build_tree_scalar, excluded=['N', 'option_type', 'american'])
        return v_tree(S, K, T, r, sigma, q, N=N, option_type=option_type, american=american)
    return _build_tree_scalar(S, K, T, r, sigma, q, N, option_type, american)
