import numpy as np
from math import exp, log, sqrt, pi
from typing import Union
from optionpricer.core import OptionContract, MarketState, OptionType
from optionpricer.models.heston import _heston_cf, _GL_NODES_64, _GL_WEIGHTS_64


def _bates_cf(u, S, r, q, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j):
    """Bates (1996) SVJD characteristic function.

    Extends the Heston CF with a compound Poisson jump component where
    log-jump sizes are normally distributed.
    """
    kappa_bar = exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    iu = 1j * u
    jump_cf = np.exp(
        lam * T * (np.exp(iu * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1.0 - iu * kappa_bar)
    )
    return _heston_cf(u, S, r, q, T, v0, kappa, theta, sigma_v, rho) * jump_cf


def bates_price(contract: OptionContract, market: MarketState,
                v0: float, kappa: float, theta: float,
                sigma_v: float, rho: float, lam: float = 0.1,
                mu_j: float = -0.05, sigma_j: float = 0.1) -> Union[float, np.ndarray]:
    """Price a European option under the Bates SVJD model.

    Combines Heston stochastic volatility with Merton-style log-normal jumps.
    Uses Gauss-Laguerre quadrature for the characteristic function integral.

    Args:
        contract: European option contract.
        market: Market state.
        v0: Initial instantaneous variance.
        kappa: Variance mean-reversion speed.
        theta: Long-run variance.
        sigma_v: Vol-of-vol.
        rho: Asset-variance correlation.
        lam: Jump intensity (jumps per year).
        mu_j: Mean of log-jump size.
        sigma_j: Std dev of log-jump size.

    Returns:
        Option price as float or ndarray.
    """
    S = float(market.spot)
    K_arr = np.atleast_1d(np.asarray(contract.strike, dtype=np.float64))
    T = float(contract.expiry)
    r = float(market.rate)
    q = float(market.dividend)
    is_call = contract.option_type == OptionType.CALL

    prices = np.empty(K_arr.shape[0])
    nodes = _GL_NODES_64
    weights = _GL_WEIGHTS_64

    for idx, K in enumerate(K_arr):
        k = log(K / S)
        cf_vals = _bates_cf(nodes - 0.5j, S, r, q, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
        integrand = np.real(np.exp(-1j * nodes * k) * cf_vals / (nodes ** 2 + 0.25))
        integral = np.sum(weights * np.exp(nodes) * integrand)
        call = S * exp(-q * T) - (sqrt(S * K) * exp(-(r + q) * T / 2.0) / pi) * integral

        if is_call:
            prices[idx] = max(call, 0.0)
        else:
            prices[idx] = max(call - S * exp(-q * T) + K * exp(-r * T), 0.0)

    if K_arr.shape[0] == 1 and np.ndim(contract.strike) == 0:
        return float(prices[0])
    return prices


def bates_characteristic(u, S, r, q, sigma, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j):
    """Bates CF compatible with carr_madan_fft interface."""
    return _bates_cf(u, S, r, q, T, v0, kappa, theta, sigma_v, rho, lam, mu_j, sigma_j)
