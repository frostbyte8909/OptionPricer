import numpy as np
from math import erfc, sqrt, exp, log
from typing import Union
from optionpricer.core import OptionContract, MarketState, OptionType


_SQRT2 = sqrt(2.0)
_INV_SQRT2PI = 1.0 / sqrt(2.0 * np.pi)


def _phi(x):
    return 0.5 * erfc(-x / _SQRT2)


def _npdf(x):
    return _INV_SQRT2PI * exp(-0.5 * x * x)


def _bsm_price_and_greeks(S, K, T, r, q, sigma, is_call):
    sqT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT

    df_q = exp(-q * T)
    df_r = exp(-r * T)
    nd1 = _npdf(d1)

    if is_call:
        price = S * df_q * _phi(d1) - K * df_r * _phi(d2)
    else:
        price = K * df_r * _phi(-d2) - S * df_q * _phi(-d1)

    vega = S * df_q * nd1 * sqT
    vanna = -df_q * nd1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return price, vega, vanna, volga


def vanna_volga_price(contract: OptionContract, market: MarketState,
                      K1: float, K2: float, K3: float,
                      sigma1: float, sigma_atm: float,
                      sigma3: float) -> Union[float, np.ndarray]:
    """Price a first-generation exotic using the Vanna-Volga method.

    Adjusts the BSM price by the cost of replicating the smile risk using
    three liquid market instruments at strikes K1 (25-delta put), K2 (ATM),
    and K3 (25-delta call).

    Args:
        contract: Option contract.
        market: Market state (vol field is used as the ATM flat vol for BSM).
        K1: 25-delta put strike (lower pillar).
        K2: ATM strike (middle pillar).
        K3: 25-delta call strike (upper pillar).
        sigma1: Market implied vol at K1.
        sigma_atm: Market implied vol at K2 (ATM).
        sigma3: Market implied vol at K3.

    Returns:
        Vanna-Volga adjusted option price.
    """
    S = float(market.spot)
    K = float(contract.strike)
    T = float(contract.expiry)
    r = float(market.rate)
    q = float(market.dividend)
    is_call = contract.option_type == OptionType.CALL

    V_bs, vega_x, vanna_x, volga_x = _bsm_price_and_greeks(
        S, K, T, r, q, sigma_atm, is_call)

    _, vega_1, vanna_1, volga_1 = _bsm_price_and_greeks(
        S, K1, T, r, q, sigma_atm, True)
    _, vega_2, vanna_2, volga_2 = _bsm_price_and_greeks(
        S, K2, T, r, q, sigma_atm, True)
    _, vega_3, vanna_3, volga_3 = _bsm_price_and_greeks(
        S, K3, T, r, q, sigma_atm, True)

    A = np.array([
        [vega_1, vega_2, vega_3],
        [vanna_1, vanna_2, vanna_3],
        [volga_1, volga_2, volga_3],
    ])

    b_vec = np.array([vega_x, vanna_x, volga_x])

    try:
        weights = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        return V_bs

    cost_1 = _bsm_price_and_greeks(S, K1, T, r, q, sigma1, True)[0] - \
             _bsm_price_and_greeks(S, K1, T, r, q, sigma_atm, True)[0]
    cost_2 = _bsm_price_and_greeks(S, K2, T, r, q, sigma_atm, True)[0] - \
             _bsm_price_and_greeks(S, K2, T, r, q, sigma_atm, True)[0]
    cost_3 = _bsm_price_and_greeks(S, K3, T, r, q, sigma3, True)[0] - \
             _bsm_price_and_greeks(S, K3, T, r, q, sigma_atm, True)[0]

    adjustment = weights[0] * cost_1 + weights[1] * cost_2 + weights[2] * cost_3

    return V_bs + adjustment
