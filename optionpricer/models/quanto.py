import numpy as np
from math import erfc, sqrt, exp, log
from typing import Union
from optionpricer.core import OptionContract, MarketState, OptionType


_SQRT2 = sqrt(2.0)


def _phi(x):
    return 0.5 * erfc(-x / _SQRT2)


def quanto_price(contract: OptionContract, market: MarketState,
                 r_domestic: float, sigma_fx: float,
                 rho: float, fx_rate: float = 1.0) -> Union[float, np.ndarray]:
    """Price a European quanto option.

    A quanto option pays in a currency different from the underlying asset's
    denomination. The quanto adjustment modifies the drift by the correlation
    between the asset and the FX rate.

    Args:
        contract: European option contract (strike in domestic currency).
        market: Market state where rate is the foreign risk-free rate.
        r_domestic: Domestic (payout) currency risk-free rate.
        sigma_fx: Volatility of the FX rate.
        rho: Correlation between asset returns and FX returns.
        fx_rate: Current FX rate (units of domestic per unit of foreign).

    Returns:
        Option price in domestic currency.
    """
    S = np.atleast_1d(np.asarray(market.spot, dtype=np.float64))
    K = np.atleast_1d(np.asarray(contract.strike, dtype=np.float64))
    T = float(contract.expiry)
    r_f = float(market.rate)
    sigma = float(market.volatility)
    q = float(market.dividend)
    is_call = contract.option_type == OptionType.CALL

    r_quanto = r_f - rho * sigma * sigma_fx
    b = np.broadcast(S, K)
    S_bc = np.broadcast_to(S, b.shape).ravel()
    K_bc = np.broadcast_to(K, b.shape).ravel()

    sqT = sqrt(T)
    d1 = (np.log(S_bc / K_bc) + (r_quanto - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT

    from scipy.special import erfc as erfc_vec
    Nd1 = 0.5 * erfc_vec(-d1 / _SQRT2)
    Nd2 = 0.5 * erfc_vec(-d2 / _SQRT2)

    df = exp(-r_domestic * T)
    df_q = np.exp(-q * T)
    fwd = S_bc * np.exp((r_quanto - q) * T)

    if is_call:
        prices = fx_rate * df * (fwd * Nd1 - K_bc * Nd2)
    else:
        Nnd1 = 0.5 * erfc_vec(d1 / _SQRT2)
        Nnd2 = 0.5 * erfc_vec(d2 / _SQRT2)
        prices = fx_rate * df * (K_bc * Nnd2 - fwd * Nnd1)

    prices = np.maximum(prices, 0.0)

    is_scalar = np.ndim(market.spot) == 0 and np.ndim(contract.strike) == 0
    if is_scalar:
        return float(prices[0])
    return prices.reshape(b.shape)
