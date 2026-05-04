import numpy as np
from math import erfc, sqrt, exp, log
from numba import njit, prange
from typing import Union, Dict, Tuple
from optionpricer.core import OptionContract, MarketState, OptionType


_SQRT2 = sqrt(2.0)
_INV_SQRT2PI = 1.0 / sqrt(2.0 * np.pi)


def aad_greeks(contract: OptionContract, market: MarketState) -> Dict[str, Union[float, np.ndarray]]:
    """Compute first and second order Greeks via closed-form AAD.

    Evaluates the full gradient vector (Delta, Gamma, Vega, Theta, Rho)
    and second-order cross-Greeks (Vanna, Volga/Vomma, Charm) in a single
    analytical pass using the BSM adjoint formulation.

    Args:
        contract: European option contract.
        market: Market state.

    Returns:
        Dict with keys: delta, gamma, vega, theta, rho, vanna, volga, charm.
    """
    S = float(market.spot)
    K = float(contract.strike)
    T = float(contract.expiry)
    r = float(market.rate)
    sigma = float(market.volatility)
    q = float(market.dividend)
    is_call = contract.option_type == OptionType.CALL

    sqT = sqrt(T)
    d1 = (log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqT)
    d2 = d1 - sigma * sqT

    phi_d1 = 0.5 * erfc(-d1 / _SQRT2)
    phi_d2 = 0.5 * erfc(-d2 / _SQRT2)
    n_d1 = _INV_SQRT2PI * exp(-0.5 * d1 * d1)

    df_q = exp(-q * T)
    df_r = exp(-r * T)

    if is_call:
        delta = df_q * phi_d1
        theta = (-df_q * S * n_d1 * sigma / (2.0 * sqT)
                 - r * K * df_r * phi_d2
                 + q * S * df_q * phi_d1)
        rho = K * T * df_r * phi_d2
    else:
        delta = -df_q * (1.0 - phi_d1)
        phi_neg_d2 = 1.0 - phi_d2
        theta = (-df_q * S * n_d1 * sigma / (2.0 * sqT)
                 + r * K * df_r * phi_neg_d2
                 - q * S * df_q * (1.0 - phi_d1))
        rho = -K * T * df_r * phi_neg_d2

    gamma = df_q * n_d1 / (S * sigma * sqT)
    vega = S * df_q * n_d1 * sqT
    vanna = -df_q * n_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    charm = df_q * n_d1 * (2.0 * (r - q) * T - d2 * sigma * sqT) / (2.0 * T * sigma * sqT)

    return {
        "delta": delta, "gamma": gamma, "vega": vega,
        "theta": theta, "rho": rho, "vanna": vanna,
        "volga": volga, "charm": charm,
    }


@njit(parallel=True, fastmath=True, cache=True)
def _malliavin_delta_kernel(payoffs, W_T, S0, sigma, T):
    """Numba kernel for Malliavin-weighted Delta."""
    N = payoffs.shape[0]
    acc = 0.0
    for i in prange(N):
        acc += payoffs[i] * W_T[i] / (S0 * sigma * T)
    return acc / N


@njit(parallel=True, fastmath=True, cache=True)
def _malliavin_gamma_kernel(payoffs, W_T, S0, sigma, T):
    """Numba kernel for Malliavin-weighted Gamma."""
    N = payoffs.shape[0]
    acc = 0.0
    inv_S2 = 1.0 / (S0 * S0 * sigma * sigma * T)
    for i in prange(N):
        hw = (W_T[i] ** 2 - sigma * T * W_T[i] - T) / (sigma * T)
        acc += payoffs[i] * hw * inv_S2
    return acc / N


@njit(parallel=True, fastmath=True, cache=True)
def _malliavin_vega_kernel(payoffs, W_T, S0, sigma, T):
    """Numba kernel for Malliavin-weighted Vega."""
    N = payoffs.shape[0]
    acc = 0.0
    for i in prange(N):
        hw = (W_T[i] ** 2 - T) / sigma - W_T[i]
        acc += payoffs[i] * hw
    return acc / N


def malliavin_greeks(payoffs: np.ndarray, W_T: np.ndarray,
                     S0: float, sigma: float, T: float,
                     r: float) -> Dict[str, float]:
    """Compute Monte Carlo Greeks using Malliavin calculus weight functions.

    Produces smooth, unbiased Greek estimates even for non-differentiable
    payoffs (digitals, barriers) where finite-difference bumping fails.

    Args:
        payoffs: Array of discounted MC payoffs, shape (N_paths,).
        W_T: Terminal Brownian motion values, shape (N_paths,).
        S0: Initial spot price.
        sigma: Volatility used in simulation.
        T: Time to expiry.
        r: Risk-free rate.

    Returns:
        Dict with keys: delta, gamma, vega.
    """
    df = exp(-r * T)
    delta = df * _malliavin_delta_kernel(payoffs, W_T, S0, sigma, T)
    gamma = df * _malliavin_gamma_kernel(payoffs, W_T, S0, sigma, T)
    vega = df * _malliavin_vega_kernel(payoffs, W_T, S0, sigma, T)

    return {"delta": delta, "gamma": gamma, "vega": vega}
