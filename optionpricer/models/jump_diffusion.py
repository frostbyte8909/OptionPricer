import numpy as np
from math import erfc, sqrt, exp, log, factorial
from typing import Union
from optionpricer.core import OptionContract, MarketState, OptionType


_SQRT2 = sqrt(2.0)


def _phi(x: float) -> float:
    return 0.5 * erfc(-x / _SQRT2)


def merton_jump_diffusion(contract: OptionContract, market: MarketState,
                          lam: float = 0.1, mu_j: float = -0.05,
                          sigma_j: float = 0.1,
                          n_terms: int = 50) -> Union[float, np.ndarray]:
    """Price a European option under the Merton Jump-Diffusion model.

    The asset follows GBM with superimposed compound Poisson jumps
    where jump sizes are log-normally distributed. The price is expressed
    as an infinite series of Black-Scholes prices weighted by Poisson
    probabilities.

    Args:
        contract: Option contract specification.
        market: Market state (spot, rate, vol, dividend).
        lam: Jump intensity (expected jumps per year).
        mu_j: Mean of log-jump size.
        sigma_j: Standard deviation of log-jump size.
        n_terms: Number of Poisson series terms (50 is typically sufficient
            for convergence to machine precision).

    Returns:
        Option price as float (scalar inputs) or ndarray.
    """
    S = np.atleast_1d(np.asarray(market.spot, dtype=np.float64))
    K = np.atleast_1d(np.asarray(contract.strike, dtype=np.float64))
    T = float(contract.expiry)
    r = float(market.rate)
    sigma = float(market.volatility)
    q = float(market.dividend)
    is_call = contract.option_type == OptionType.CALL

    kappa = exp(mu_j + 0.5 * sigma_j ** 2) - 1.0
    lam_prime = lam * (1.0 + kappa)

    b = np.broadcast(S, K)
    S_bc = np.broadcast_to(S, b.shape).ravel()
    K_bc = np.broadcast_to(K, b.shape).ravel()
    result = np.zeros(S_bc.shape[0])

    log_weight = -lam_prime * T
    for n in range(n_terms):
        sigma_n = sqrt(sigma ** 2 + n * sigma_j ** 2 / T)
        r_n = r - lam * kappa + n * (mu_j + 0.5 * sigma_j ** 2) / T

        if n > 0:
            log_weight += log(lam_prime * T) - log(n)

        w = exp(log_weight)
        if w < 1e-16 and n > 5:
            break

        for i in range(S_bc.shape[0]):
            sqT = sqrt(T)
            d1 = (log(S_bc[i] / K_bc[i]) + (r_n - q + 0.5 * sigma_n ** 2) * T) / (sigma_n * sqT)
            d2 = d1 - sigma_n * sqT
            df_r = exp(-r_n * T)
            df_q = exp(-q * T)

            if is_call:
                bs_n = S_bc[i] * df_q * _phi(d1) - K_bc[i] * df_r * _phi(d2)
            else:
                bs_n = K_bc[i] * df_r * _phi(-d2) - S_bc[i] * df_q * _phi(-d1)

            result[i] += w * bs_n

    is_scalar = (np.ndim(market.spot) == 0 and np.ndim(contract.strike) == 0)
    if is_scalar:
        return float(result[0])
    return result.reshape(b.shape)
