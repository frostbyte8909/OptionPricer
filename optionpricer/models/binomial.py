import numpy as np
from numba import njit
from typing import Union
from optionpricer.core import OptionContract, MarketState

try:
    from optionpricer.models._binomial_cy import _build_tree_cython_vectorized as _build_tree_core
except ImportError:
    @njit(cache=True, fastmath=True)
    def _build_tree_core(S_arr, K_arr, result_arr, u, d, df, p, N, is_call, american):
        num_options = S_arr.shape[0]
        option = np.empty(N + 1)
        for k in range(num_options):
            for j in range(N + 1):
                S_ij = S_arr[k] * (u ** j) * (d ** (N - j))
                if is_call:
                    option[j] = S_ij - K_arr[k] if S_ij > K_arr[k] else 0.0
                else:
                    option[j] = K_arr[k] - S_ij if K_arr[k] > S_ij else 0.0
                    
            for i in range(N - 1, -1, -1):
                for j in range(i + 1):
                    option[j] = df * (p * option[j + 1] + (1 - p) * option[j])
                    if american:
                        S_ij = S_arr[k] * (u ** j) * (d ** (i - j))
                        if is_call:
                            intrinsic = S_ij - K_arr[k] if S_ij > K_arr[k] else 0.0
                        else:
                            intrinsic = K_arr[k] - S_ij if K_arr[k] > S_ij else 0.0
                        if intrinsic > option[j]:
                            option[j] = intrinsic
            result_arr[k] = option[0]

def build_tree(contract: OptionContract, market: MarketState, N: int = 1000) -> Union[float, np.ndarray]:
    b = np.broadcast(market.spot, contract.strike)
    S_arr = np.broadcast_to(market.spot, b.shape).astype(float).ravel()
    K_arr = np.broadcast_to(contract.strike, b.shape).astype(float).ravel()
    
    T = float(contract.expiry)
    r = float(market.rate)
    sigma = float(market.volatility)
    q = float(market.dividend)
    
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    df = np.exp(-r * dt)
    p  = (np.exp((r - q) * dt) - d) / (u - d)

    result_arr = np.empty_like(S_arr)
    _build_tree_core(S_arr, K_arr, result_arr, float(u), float(d), float(df), float(p), int(N), contract.option_type == "call", contract.american)
    
    result = result_arr.reshape(b.shape)
    if result.ndim == 0:
        return float(result.item())
    return result
