import numpy as np
from scipy.stats import norm
from typing import Union
from optionpricer.core import OptionContract, MarketState

def black_scholes(contract: OptionContract, market: MarketState) -> Union[float, np.ndarray]:
    b = np.broadcast(market.spot, contract.strike, contract.expiry, market.rate, market.volatility, market.dividend)
    S_arr = np.broadcast_to(market.spot, b.shape).astype(float)
    K_arr = np.broadcast_to(contract.strike, b.shape).astype(float)
    T_arr = np.broadcast_to(contract.expiry, b.shape).astype(float)
    r_arr = np.broadcast_to(market.rate, b.shape).astype(float)
    sig_arr = np.broadcast_to(market.volatility, b.shape).astype(float)
    q_arr = np.broadcast_to(market.dividend, b.shape).astype(float)

    is_expired = T_arr <= 1e-8
    is_deterministic = sig_arr <= 1e-8
    mask_edge = is_expired | is_deterministic

    result = np.zeros(b.shape, dtype=float)

    if np.any(mask_edge):
        if contract.option_type == "call":
            result[mask_edge] = np.maximum(S_arr[mask_edge] * np.exp(-q_arr[mask_edge] * T_arr[mask_edge]) - K_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]), 0.0)
        elif contract.option_type == "put":
            result[mask_edge] = np.maximum(K_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]) - S_arr[mask_edge] * np.exp(-q_arr[mask_edge] * T_arr[mask_edge]), 0.0)

    mask_calc = ~mask_edge
    if np.any(mask_calc):
        S_c, K_c, T_c, r_c, sig_c, q_c = S_arr[mask_calc], K_arr[mask_calc], T_arr[mask_calc], r_arr[mask_calc], sig_arr[mask_calc], q_arr[mask_calc]
        d1 = (np.log(S_c / K_c) + (r_c - q_c + 0.5 * sig_c**2) * T_c) / (sig_c * np.sqrt(T_c))
        d2 = d1 - sig_c * np.sqrt(T_c)
        df_r = np.exp(-r_c * T_c)
        df_q = np.exp(-q_c * T_c)

        if contract.option_type == "call":
            result[mask_calc] = S_c * df_q * norm.cdf(d1) - K_c * df_r * norm.cdf(d2)
        else:
            result[mask_calc] = K_c * df_r * norm.cdf(-d2) - S_c * df_q * norm.cdf(-d1)

    if result.ndim == 0:
        return float(result.item())
    return result
