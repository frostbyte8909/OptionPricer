import numpy as np
from numba import njit
from scipy.optimize import minimize
from typing import Tuple


@njit(cache=True, fastmath=True)
def _garch_variance_path(returns: np.ndarray, omega: float, alpha: float, beta: float) -> np.ndarray:
    """Compute the conditional variance path for GARCH(1,1).

    Args:
        returns: Array of log-returns.
        omega: Long-run variance weight.
        alpha: Shock coefficient.
        beta: Persistence coefficient.

    Returns:
        Array of conditional variances at each timestep.
    """
    T = returns.shape[0]
    sigma2 = np.empty(T)
    sigma2[0] = omega / (1.0 - alpha - beta)

    for t in range(1, T):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        if sigma2[t] < 1e-12:
            sigma2[t] = 1e-12

    return sigma2


@njit(cache=True, fastmath=True)
def _garch_neg_loglik(returns: np.ndarray, omega: float, alpha: float, beta: float) -> float:
    """Negative log-likelihood of the GARCH(1,1) model.

    Args:
        returns: Array of log-returns.
        omega: Long-run variance weight.
        alpha: Shock coefficient.
        beta: Persistence coefficient.

    Returns:
        Scalar negative log-likelihood value.
    """
    sigma2 = _garch_variance_path(returns, omega, alpha, beta)
    T = returns.shape[0]
    nll = 0.0
    for t in range(T):
        nll += np.log(sigma2[t]) + returns[t] ** 2 / sigma2[t]
    return 0.5 * nll


def garch_fit(prices: np.ndarray) -> Tuple[float, float, float, float]:
    """Calibrate GARCH(1,1) parameters via MLE on price data.

    Args:
        prices: Array of asset prices (chronologically ordered).

    Returns:
        Tuple of (omega, alpha, beta, sigma_forecast) where sigma_forecast
        is the annualized one-step-ahead volatility.
    """
    returns = np.diff(np.log(prices))
    sample_var = np.var(returns)

    def objective(params):
        omega, alpha, beta = params
        if alpha + beta >= 0.9999 or omega <= 0 or alpha < 0 or beta < 0:
            return 1e12
        return _garch_neg_loglik(returns, omega, alpha, beta)

    x0 = np.array([sample_var * 0.05, 0.08, 0.88])
    bounds = [(1e-8, sample_var), (1e-8, 0.5), (0.5, 0.9999)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    omega, alpha, beta = result.x

    sigma2_path = _garch_variance_path(returns, omega, alpha, beta)
    h_next = omega + alpha * returns[-1] ** 2 + beta * sigma2_path[-1]
    sigma_forecast = np.sqrt(h_next * 252)

    return omega, alpha, beta, sigma_forecast


def ewma_volatility(prices: np.ndarray, span: int = 30) -> float:
    """Exponentially Weighted Moving Average volatility fallback.

    Args:
        prices: Array of asset prices.
        span: EWMA decay span in trading days.

    Returns:
        Annualized EWMA volatility estimate.
    """
    returns = np.diff(np.log(prices))
    lam = 1.0 - 2.0 / (span + 1)
    T = returns.shape[0]

    weights = np.empty(T)
    weights[T - 1] = 1.0
    for i in range(T - 2, -1, -1):
        weights[i] = weights[i + 1] * lam
    weights /= weights.sum()

    var = np.sum(weights * returns ** 2)
    return np.sqrt(var * 252)


@njit(cache=True, fastmath=True)
def _sabr_vol_kernel(F: float, K: float, T: float, alpha: float,
                     beta: float, rho: float, nu: float) -> float:
    """Hagan's SABR implied volatility approximation kernel.

    Args:
        F: Forward price.
        K: Strike price.
        T: Time to expiry.
        alpha: Initial volatility level.
        beta: CEV exponent.
        rho: Correlation between asset and vol Brownians.
        nu: Volatility of volatility.

    Returns:
        SABR implied Black volatility for the given strike.
    """
    if abs(F - K) < 1e-12:
        FK_mid = F
        A = alpha / FK_mid ** (1.0 - beta)
        B1 = ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / FK_mid ** (2.0 - 2.0 * beta)
        B2 = 0.25 * rho * beta * nu * alpha / FK_mid ** (1.0 - beta)
        B3 = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
        return A * (1.0 + (B1 + B2 + B3) * T)

    FK = F * K
    FK_beta = FK ** ((1.0 - beta) / 2.0)
    log_FK = np.log(F / K)

    denom = FK_beta * (1.0 + (1.0 - beta) ** 2 / 24.0 * log_FK ** 2
                       + (1.0 - beta) ** 4 / 1920.0 * log_FK ** 4)

    z = (nu / alpha) * FK_beta * log_FK
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z ** 2)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))

    if abs(x_z) < 1e-12:
        zeta = 1.0
    else:
        zeta = z / x_z

    B1 = ((1.0 - beta) ** 2 / 24.0) * alpha ** 2 / FK ** (1.0 - beta)
    B2 = 0.25 * rho * beta * nu * alpha / FK_beta
    B3 = (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2

    return (alpha / denom) * zeta * (1.0 + (B1 + B2 + B3) * T)


def sabr_implied_vol(F: float, K: np.ndarray, T: float, alpha: float,
                     beta: float, rho: float, nu: float) -> np.ndarray:
    """Vectorized SABR implied volatility across a strike array.

    Args:
        F: Forward price.
        K: Array of strike prices.
        T: Time to expiry.
        alpha: SABR alpha parameter.
        beta: SABR beta (CEV exponent, typically 0.5 for rates, 1.0 for FX).
        rho: SABR rho (vol-spot correlation).
        nu: SABR nu (vol-of-vol).

    Returns:
        Array of implied Black volatilities for each strike.
    """
    K_arr = np.atleast_1d(np.asarray(K, dtype=np.float64))
    out = np.empty_like(K_arr)
    for i in range(K_arr.shape[0]):
        out[i] = _sabr_vol_kernel(F, K_arr[i], T, alpha, beta, rho, nu)
    return out
