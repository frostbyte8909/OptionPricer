import numpy as np
from math import exp, log, sqrt, pi
from typing import Union
from optionpricer.core import OptionContract, MarketState, OptionType


_GL_NODES_64, _GL_WEIGHTS_64 = np.polynomial.laguerre.laggauss(64)


def _heston_cf(u, S, r, q, T, v0, kappa, theta, sigma_v, rho):
    """Heston (1993) characteristic function with Albrecher stability fix."""
    iu = 1j * u
    d = np.sqrt((kappa - rho * sigma_v * iu) ** 2 + sigma_v ** 2 * (iu + u ** 2))
    g = (kappa - rho * sigma_v * iu - d) / (kappa - rho * sigma_v * iu + d)

    exp_dT = np.exp(-d * T)
    C = iu * (r - q) * T + (kappa * theta / sigma_v ** 2) * (
        (kappa - rho * sigma_v * iu - d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    D = ((kappa - rho * sigma_v * iu - d) / sigma_v ** 2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)

    return np.exp(C + D * v0 + iu * np.log(S))


def _heston_p(S, K, r, q, T, v0, kappa, theta, sigma_v, rho, phi_type):
    """Compute P1 or P2 via Gauss-Laguerre quadrature."""
    nodes = _GL_NODES_64
    weights = _GL_WEIGHTS_64
    lnK = log(K)

    integrand = np.zeros(len(nodes))
    for i, u in enumerate(nodes):
        if u < 1e-12:
            continue
        if phi_type == 1:
            b = kappa - rho * sigma_v
        else:
            b = kappa

        iu = 1j * u
        d = np.sqrt((rho * sigma_v * iu - b) ** 2 - sigma_v ** 2 * (2.0 * iu * (1 if phi_type == 1 else 0) - u ** 2))
        g = (b - rho * sigma_v * iu + d) / (b - rho * sigma_v * iu - d)

        exp_dT = np.exp(d * T)
        C = (r - q) * iu * T + (kappa * theta / sigma_v ** 2) * (
            (b - rho * sigma_v * iu + d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
        )
        D = ((b - rho * sigma_v * iu + d) / sigma_v ** 2) * (1.0 - exp_dT) / (1.0 - g * exp_dT)

        cf = np.exp(C + D * v0 + iu * log(S))
        integrand[i] = np.real(np.exp(-iu * lnK) * cf / (iu)) * np.exp(nodes[i])

    return 0.5 + (1.0 / pi) * np.sum(weights * integrand)


def heston_price(contract: OptionContract, market: MarketState,
                 v0: float, kappa: float, theta: float,
                 sigma_v: float, rho: float) -> Union[float, np.ndarray]:
    """Price a European option under the Heston stochastic volatility model.

    Uses the Gil-Pelaez inversion with Gauss-Laguerre quadrature (64 nodes)
    for the P1/P2 integrals, achieving fast convergence.

    Args:
        contract: European option contract.
        market: Market state (volatility field is ignored; v0 is used instead).
        v0: Initial instantaneous variance.
        kappa: Mean-reversion speed of variance.
        theta: Long-run variance level.
        sigma_v: Volatility of variance (vol-of-vol).
        rho: Correlation between asset and variance Brownians.

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

    for idx, K in enumerate(K_arr):
        P1 = _heston_p(S, K, r, q, T, v0, kappa, theta, sigma_v, rho, 1)
        P2 = _heston_p(S, K, r, q, T, v0, kappa, theta, sigma_v, rho, 2)

        call = S * exp(-q * T) * P1 - K * exp(-r * T) * P2

        if is_call:
            prices[idx] = max(call, 0.0)
        else:
            prices[idx] = max(call - S * exp(-q * T) + K * exp(-r * T), 0.0)

    if K_arr.shape[0] == 1 and np.ndim(contract.strike) == 0:
        return float(prices[0])
    return prices


def heston_characteristic(u, S, r, q, sigma, T, v0, kappa, theta, sigma_v, rho):
    """Heston CF compatible with carr_madan_fft interface.

    Args:
        u: Fourier variable array.
        S: Spot price.
        r: Risk-free rate.
        q: Dividend yield.
        sigma: Ignored (present for interface compatibility).
        T: Time to expiry.
        v0: Initial variance.
        kappa: Mean-reversion speed.
        theta: Long-run variance.
        sigma_v: Vol-of-vol.
        rho: Correlation.

    Returns:
        Complex array of CF values.
    """
    return _heston_cf(u, S, r, q, T, v0, kappa, theta, sigma_v, rho)
