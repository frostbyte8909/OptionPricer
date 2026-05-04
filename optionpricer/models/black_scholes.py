import numpy as np
from math import erfc, sqrt
from typing import Union, Tuple, Dict
from optionpricer.core import OptionContract, MarketState, OptionType


_SQRT2 = sqrt(2.0)
_INV_SQRT2PI = 1.0 / sqrt(2.0 * np.pi)


def _phi_scalar(x: float) -> float:
    return 0.5 * erfc(-x / _SQRT2)


def _phi_vec(x: np.ndarray) -> np.ndarray:
    from scipy.special import erfc as erfc_vec
    return 0.5 * erfc_vec(-x / _SQRT2)


def _npdf_vec(x: np.ndarray) -> np.ndarray:
    return _INV_SQRT2PI * np.exp(-0.5 * x * x)


def black_scholes(contract: OptionContract, market: MarketState, return_greeks: bool = False) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], Dict[str, Union[float, np.ndarray]]]]:
    """Compute Black-Scholes-Merton European option prices.

    Uses the complementary error function (erfc) for the normal CDF to
    maintain machine precision at extreme tail values (|d| > 8) where
    scipy.stats.norm.cdf loses accuracy.

    Args:
        contract: Option contract specification.
        market: Market state with spot, rate, volatility, and dividend.
        return_greeks: If True, also return the full Greek vector.

    Returns:
        Option price(s). If return_greeks is True, returns a tuple of
        (price, greeks_dict).
    """
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

    if return_greeks:
        delta = np.zeros(b.shape, dtype=float)
        gamma = np.zeros(b.shape, dtype=float)
        vega = np.zeros(b.shape, dtype=float)
        theta = np.zeros(b.shape, dtype=float)
        rho = np.zeros(b.shape, dtype=float)

    if np.any(mask_edge):
        if contract.option_type == OptionType.CALL:
            result[mask_edge] = np.maximum(S_arr[mask_edge] * np.exp(-q_arr[mask_edge] * T_arr[mask_edge]) - K_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]), 0.0)
            if return_greeks:
                delta[mask_edge] = np.where(S_arr[mask_edge] > K_arr[mask_edge], np.exp(-q_arr[mask_edge] * T_arr[mask_edge]), 0.0)
                rho[mask_edge] = np.where(S_arr[mask_edge] > K_arr[mask_edge], K_arr[mask_edge] * T_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]), 0.0)
        elif contract.option_type == OptionType.PUT:
            result[mask_edge] = np.maximum(K_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]) - S_arr[mask_edge] * np.exp(-q_arr[mask_edge] * T_arr[mask_edge]), 0.0)
            if return_greeks:
                delta[mask_edge] = np.where(S_arr[mask_edge] < K_arr[mask_edge], -np.exp(-q_arr[mask_edge] * T_arr[mask_edge]), 0.0)
                rho[mask_edge] = np.where(S_arr[mask_edge] < K_arr[mask_edge], -K_arr[mask_edge] * T_arr[mask_edge] * np.exp(-r_arr[mask_edge] * T_arr[mask_edge]), 0.0)

    mask_calc = ~mask_edge
    if np.any(mask_calc):
        S_c, K_c, T_c, r_c, sig_c, q_c = S_arr[mask_calc], K_arr[mask_calc], T_arr[mask_calc], r_arr[mask_calc], sig_arr[mask_calc], q_arr[mask_calc]
        d1 = (np.log(S_c / K_c) + (r_c - q_c + 0.5 * sig_c**2) * T_c) / (sig_c * np.sqrt(T_c))
        d2 = d1 - sig_c * np.sqrt(T_c)
        df_r = np.exp(-r_c * T_c)
        df_q = np.exp(-q_c * T_c)

        Nd1 = _phi_vec(d1)
        Nd2 = _phi_vec(d2)
        nd1 = _npdf_vec(d1)

        if contract.option_type == OptionType.CALL:
            result[mask_calc] = S_c * df_q * Nd1 - K_c * df_r * Nd2
            if return_greeks:
                delta[mask_calc] = df_q * Nd1
                gamma[mask_calc] = df_q * nd1 / (S_c * sig_c * np.sqrt(T_c))
                vega[mask_calc] = S_c * df_q * nd1 * np.sqrt(T_c)
                theta[mask_calc] = -df_q * S_c * nd1 * sig_c / (2 * np.sqrt(T_c)) - r_c * K_c * df_r * Nd2 + q_c * S_c * df_q * Nd1
                rho[mask_calc] = K_c * T_c * df_r * Nd2
        else:
            Nnd1 = _phi_vec(-d1)
            Nnd2 = _phi_vec(-d2)
            result[mask_calc] = K_c * df_r * Nnd2 - S_c * df_q * Nnd1
            if return_greeks:
                delta[mask_calc] = -df_q * Nnd1
                gamma[mask_calc] = df_q * nd1 / (S_c * sig_c * np.sqrt(T_c))
                vega[mask_calc] = S_c * df_q * nd1 * np.sqrt(T_c)
                theta[mask_calc] = -df_q * S_c * nd1 * sig_c / (2 * np.sqrt(T_c)) + r_c * K_c * df_r * Nnd2 - q_c * S_c * df_q * Nnd1
                rho[mask_calc] = -K_c * T_c * df_r * Nnd2

    if result.ndim == 0:
        result = float(result.item())
        if return_greeks:
            return result, {
                "delta": float(delta.item()),
                "gamma": float(gamma.item()),
                "vega": float(vega.item()),
                "theta": float(theta.item()),
                "rho": float(rho.item())
            }
        return result

    if return_greeks:
        return result, {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho
        }
    return result
