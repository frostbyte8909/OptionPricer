import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from typing import Union

def _monte_carlo_scalar(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0, N: int = 32_768, option_type: str = "call", seed: int = None) -> float:
    half = N // 2
    half = 1 << (half - 1).bit_length()

    sampler = Sobol(d=1, scramble=True, seed=seed)
    Z = norm.ppf(np.clip(sampler.random(half), 1e-10, 1 - 1e-10)).ravel()

    drift  = (r - q - 0.5 * sigma**2) * T
    vol    = sigma * np.sqrt(T)
    
    ST_all = np.empty(N)
    ST_all[:half] = S * np.exp(drift + vol * Z)
    ST_all[half:] = S * np.exp(drift - vol * Z)

    p_all = np.empty(N)
    if option_type == "call":
        np.maximum(ST_all - K, 0, out=p_all)
    else:
        np.maximum(K - ST_all, 0, out=p_all)

    E_ST   = S * np.exp((r - q) * T)
    ST_dev = ST_all - ST_all.mean()
    p_mean = p_all.mean()
    
    dot_dev = np.dot(ST_dev, ST_dev)
    if dot_dev == 0:
        c = 0.0
    else:
        c = -np.dot(p_all - p_mean, ST_dev) / dot_dev

    return np.exp(-r * T) * (p_all + c * (ST_all - E_ST)).mean()

def monte_carlo_prices(S: Union[float, np.ndarray], K: Union[float, np.ndarray], T: Union[float, np.ndarray], r: Union[float, np.ndarray], sigma: Union[float, np.ndarray], q: Union[float, np.ndarray] = 0.0, N: int = 32_768, option_type: str = "call", seed: int = None) -> Union[float, np.ndarray]:
    if any(isinstance(x, np.ndarray) for x in (S, K, T, r, sigma, q)):
        v_mc = np.vectorize(_monte_carlo_scalar, excluded=['N', 'option_type', 'seed'])
        return v_mc(S, K, T, r, sigma, q, N=N, option_type=option_type, seed=seed)
    return _monte_carlo_scalar(S, K, T, r, sigma, q, N, option_type, seed)
