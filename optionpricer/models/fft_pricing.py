import numpy as np
from typing import Union, Callable
from optionpricer.core import OptionContract, MarketState, OptionType


def _bsm_characteristic(u, S, r, q, sigma, T):
    """BSM log-normal characteristic function."""
    mu = np.log(S) + (r - q - 0.5 * sigma ** 2) * T
    return np.exp(1j * u * mu - 0.5 * sigma ** 2 * T * u ** 2)


def _merton_characteristic(u, S, r, q, sigma, T, lam, mu_j, sigma_j):
    """Merton Jump-Diffusion characteristic function."""
    kappa = np.exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    mu = np.log(S) + (r - q - 0.5 * sigma ** 2 - lam * kappa) * T
    phi_diff = np.exp(1j * u * mu - 0.5 * sigma ** 2 * T * u ** 2)
    phi_jump = np.exp(lam * T * (np.exp(1j * u * mu_j - 0.5 * sigma_j ** 2 * u ** 2) - 1.0))
    return phi_diff * phi_jump


def carr_madan_fft(contract: OptionContract, market: MarketState,
                   char_fn: Callable = None, N: int = 4096,
                   alpha: float = 1.5, eta: float = 0.25,
                   **char_kwargs) -> Union[float, np.ndarray]:
    """Price European options via the Carr-Madan FFT method.

    Args:
        contract: Option contract specification.
        market: Market state.
        char_fn: Characteristic function. Defaults to BSM.
        N: FFT grid size (power of 2).
        alpha: Dampening coefficient.
        eta: Frequency domain spacing.
        **char_kwargs: Extra kwargs for char_fn.

    Returns:
        Option price as float or ndarray.
    """
    S = float(market.spot)
    K = np.atleast_1d(np.asarray(contract.strike, dtype=np.float64))
    T = float(contract.expiry)
    r = float(market.rate)
    q = float(market.dividend)
    sigma = float(market.volatility)

    if char_fn is None:
        char_fn = _bsm_characteristic

    lam_grid = 2.0 * np.pi / (N * eta)
    b = N * lam_grid / 2.0

    v = np.arange(N) * eta
    k_grid = -b + lam_grid * np.arange(N)

    u = v - (alpha + 1.0) * 1j
    phi = char_fn(u, S, r, q, sigma, T, **char_kwargs)

    denom = alpha ** 2 + alpha - v ** 2 + 1j * (2.0 * alpha + 1.0) * v
    psi = np.exp(-r * T) * phi / denom

    simpson = 3.0 + (-1.0) ** np.arange(1, N + 1) - np.where(np.arange(N) == 0, 1.0, 0.0)
    simpson *= eta / 3.0

    x = np.exp(1j * v * b) * psi * simpson
    fft_result = np.fft.fft(x).real

    call_prices = np.exp(-alpha * k_grid) / np.pi * fft_result

    log_K = np.log(K)
    prices = np.interp(log_K, k_grid, call_prices)
    prices = np.maximum(prices, 0.0)

    if contract.option_type == OptionType.PUT:
        prices = prices - S * np.exp(-q * T) + K * np.exp(-r * T)

    if prices.ndim == 0 or (prices.ndim == 1 and prices.shape[0] == 1):
        return float(prices.ravel()[0])
    return prices
