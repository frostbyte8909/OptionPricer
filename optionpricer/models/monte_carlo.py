import numpy as np
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from typing import Union
from optionpricer.core import OptionContract, MarketState

def monte_carlo_prices(contract: OptionContract, market: MarketState, N: int = 32_768, seed: int = None) -> Union[float, np.ndarray]:
    b = np.broadcast(market.spot, contract.strike, contract.expiry, market.rate, market.volatility, market.dividend)
    S_arr = np.broadcast_to(market.spot, b.shape).astype(float).reshape(-1, 1)
    K_arr = np.broadcast_to(contract.strike, b.shape).astype(float).reshape(-1, 1)
    T_arr = np.broadcast_to(contract.expiry, b.shape).astype(float).reshape(-1, 1)
    r_arr = np.broadcast_to(market.rate, b.shape).astype(float).reshape(-1, 1)
    sig_arr = np.broadcast_to(market.volatility, b.shape).astype(float).reshape(-1, 1)
    q_arr = np.broadcast_to(market.dividend, b.shape).astype(float).reshape(-1, 1)

    half = N // 2
    half = 1 << (half - 1).bit_length()

    sampler = Sobol(d=1, scramble=True, seed=seed)
    Z = norm.ppf(np.clip(sampler.random(half), 1e-10, 1 - 1e-10)).ravel()
    Z = Z.reshape(1, -1)

    drift  = (r_arr - q_arr - 0.5 * sig_arr**2) * T_arr
    vol    = sig_arr * np.sqrt(T_arr)
    
    ST_all = np.empty((S_arr.shape[0], N))
    ST_all[:, :half] = S_arr * np.exp(drift + vol * Z)
    ST_all[:, half:] = S_arr * np.exp(drift - vol * Z)

    p_all = np.empty_like(ST_all)
    if contract.option_type == "call":
        np.maximum(ST_all - K_arr, 0, out=p_all)
    else:
        np.maximum(K_arr - ST_all, 0, out=p_all)

    E_ST   = S_arr * np.exp((r_arr - q_arr) * T_arr)
    ST_mean = ST_all.mean(axis=1, keepdims=True)
    ST_dev = ST_all - ST_mean
    p_mean = p_all.mean(axis=1, keepdims=True)
    
    dot_dev = np.sum(ST_dev * ST_dev, axis=1, keepdims=True)
    c = np.zeros_like(dot_dev)
    mask = dot_dev.ravel() != 0
    if np.any(mask):
        c[mask] = -np.sum((p_all - p_mean)[mask] * ST_dev[mask], axis=1, keepdims=True) / dot_dev[mask]

    result = np.exp(-r_arr * T_arr) * np.mean(p_all + c * (ST_all - E_ST), axis=1, keepdims=True)
    result = result.reshape(b.shape)
    
    if result.ndim == 0:
        return float(result.item())
    return result
