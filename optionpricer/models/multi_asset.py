import numpy as np
from numba import njit, prange
from typing import Union, List
from optionpricer.core import OptionType


@njit(cache=True, fastmath=True)
def _cholesky_lower(corr):
    """In-place Cholesky decomposition for correlation matrix."""
    n = corr.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s += L[i, k] * L[j, k]
            if i == j:
                L[i, j] = np.sqrt(max(corr[i, i] - s, 1e-12))
            else:
                L[i, j] = (corr[i, j] - s) / L[j, j]
    return L


@njit(parallel=True, fastmath=True, cache=True)
def _multi_asset_paths(spots, drifts, vols, L, Z, N_paths, N_steps, dt):
    """Generate correlated multi-asset GBM paths."""
    n_assets = spots.shape[0]
    payoffs_sum = np.zeros(N_paths)

    for p in prange(N_paths):
        S = spots.copy()
        for t in range(N_steps):
            corr_Z = np.zeros(n_assets)
            for i in range(n_assets):
                for j in range(i + 1):
                    corr_Z[i] += L[i, j] * Z[p, t, j]
            for i in range(n_assets):
                S[i] *= np.exp(drifts[i] * dt + vols[i] * np.sqrt(dt) * corr_Z[i])
        payoffs_sum[p] = np.mean(S)

    return payoffs_sum


def basket_option(spots: np.ndarray, vols: np.ndarray,
                  corr_matrix: np.ndarray, K: float, T: float,
                  r: float, q: np.ndarray = None,
                  option_type: str = "call",
                  weights: np.ndarray = None,
                  N_paths: int = 32768, N_steps: int = 252,
                  seed: int = None) -> float:
    """Price a basket option on multiple correlated assets via Monte Carlo.

    Uses Cholesky decomposition for correlation structure and Numba-parallel
    path generation. The basket is an arithmetic weighted average of terminal
    asset prices.

    Args:
        spots: Array of initial spot prices, shape (n_assets,).
        vols: Array of volatilities, shape (n_assets,).
        corr_matrix: Correlation matrix, shape (n_assets, n_assets).
        K: Strike price of the basket.
        T: Time to expiry.
        r: Risk-free rate.
        q: Array of dividend yields. Defaults to zero.
        option_type: 'call' or 'put'.
        weights: Portfolio weights. Defaults to equal-weight.
        N_paths: Number of MC paths.
        N_steps: Number of time steps.
        seed: Random seed.

    Returns:
        Basket option price.
    """
    spots = np.asarray(spots, dtype=np.float64)
    vols = np.asarray(vols, dtype=np.float64)
    corr_matrix = np.asarray(corr_matrix, dtype=np.float64)
    n_assets = spots.shape[0]

    if q is None:
        q = np.zeros(n_assets)
    else:
        q = np.asarray(q, dtype=np.float64)

    if weights is None:
        weights = np.ones(n_assets) / n_assets
    else:
        weights = np.asarray(weights, dtype=np.float64)

    L = _cholesky_lower(corr_matrix)
    drifts = (r - q - 0.5 * vols ** 2)
    dt = T / N_steps

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N_paths, N_steps, n_assets))

    terminal_avg = _multi_asset_paths(spots * weights, drifts, vols, L, Z, N_paths, N_steps, dt)

    df = np.exp(-r * T)
    is_call = option_type == "call"

    if is_call:
        payoffs = np.maximum(terminal_avg - K, 0.0)
    else:
        payoffs = np.maximum(K - terminal_avg, 0.0)

    return float(df * np.mean(payoffs))


def spread_option(S1: float, S2: float, vol1: float, vol2: float,
                  rho: float, K: float, T: float, r: float,
                  option_type: str = "call",
                  N_paths: int = 32768, seed: int = None) -> float:
    """Price a spread option (S1 - S2 - K) via Kirk's approximation.

    Args:
        S1: Spot price of asset 1.
        S2: Spot price of asset 2.
        vol1: Volatility of asset 1.
        vol2: Volatility of asset 2.
        rho: Correlation between assets.
        K: Strike of the spread.
        T: Time to expiry.
        r: Risk-free rate.
        option_type: 'call' or 'put'.
        N_paths: Unused (kept for API consistency).
        seed: Unused.

    Returns:
        Spread option price.
    """
    from math import erfc, sqrt, exp, log
    _SQRT2 = sqrt(2.0)

    S2_adj = S2 * np.exp(-r * T)
    F1 = S1 * np.exp(-r * T)
    ratio = S2_adj / (S2_adj + K * np.exp(-r * T))

    sigma_kirk = sqrt(vol1 ** 2 - 2 * rho * vol1 * vol2 * ratio + (vol2 * ratio) ** 2)
    F_spread = S1 / (S2 + K * np.exp(-r * T))

    d1 = (log(F_spread) + 0.5 * sigma_kirk ** 2 * T) / (sigma_kirk * sqrt(T))
    d2 = d1 - sigma_kirk * sqrt(T)

    Nd1 = 0.5 * erfc(-d1 / _SQRT2)
    Nd2 = 0.5 * erfc(-d2 / _SQRT2)
    df = exp(-r * T)

    if option_type == "call":
        return float((S2 + K * df) * (F_spread * Nd1 - Nd2) * df)
    else:
        Nnd1 = 0.5 * erfc(d1 / _SQRT2)
        Nnd2 = 0.5 * erfc(d2 / _SQRT2)
        return float((S2 + K * df) * (Nnd2 - F_spread * Nnd1) * df)
