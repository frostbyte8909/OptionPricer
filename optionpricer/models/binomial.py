import numpy as np
from numba import njit

try:
    from optionpricer.models._binomial_cy import _build_tree_cython as _build_tree_core
except ImportError:
    @njit(cache=True, fastmath=True)
    def _build_tree_core(option, S_T, u_pows, K, df, p, N, is_call, american):
        for i in range(N - 1, -1, -1):
            scalar = u_pows[N - i]
            for j in range(i + 1):
                option[j] = df * (p * option[j + 1] + (1 - p) * option[j])
                if american:
                    S_ij = S_T[j] * scalar
                    if is_call:
                        intrinsic = S_ij - K if S_ij > K else 0.0
                    else:
                        intrinsic = K - S_ij if K > S_ij else 0.0
                    if intrinsic > option[j]:
                        option[j] = intrinsic
        return option[0]


def build_tree(S: float, K: float, T: float, r: float, sigma: float, N: int = 1000, option_type: str = "call", american: bool = False) -> float:
    """
    Price an option using the Cox-Ross-Rubinstein (CRR) binomial tree model.
    Optimized internally via Cython or Numba.

    Args:
        S (float): Current asset price.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate (annualized).
        sigma (float): Volatility of the underlying asset (annualized).
        N (int, optional): Number of steps in the binomial tree. Defaults to 1000.
        option_type (str, optional): 'call' for Call option, 'put' for Put option. Defaults to 'call'.
        american (bool, optional): If True, prices an American option with early exercise. If False, prices a European option. Defaults to False.

    Returns:
        float: The theoretical price of the option.
    """
    is_call = option_type == "call"

    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    df = np.exp(-r * dt)
    p  = (np.exp(r * dt) - d) / (u - d)

    u_pows    = np.empty(N + 1)
    u_pows = np.power(u, np.arrange(N + 1))

    S_T    = S * u_pows / u_pows[::-1]
    option = np.maximum(S_T - K, 0.0) if is_call else np.maximum(K - S_T, 0.0)

    return _build_tree_core(option, S_T, u_pows, K, df, p, N, is_call, american)
