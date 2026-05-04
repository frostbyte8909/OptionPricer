import numpy as np
from scipy.optimize import minimize
from typing import Tuple


def nelson_siegel(tau: np.ndarray, beta0: float, beta1: float,
                  beta2: float, lam: float) -> np.ndarray:
    """Nelson-Siegel yield curve model.

    Args:
        tau: Array of maturities in years.
        beta0: Long-term yield level.
        beta1: Short-term component (slope).
        beta2: Medium-term component (curvature).
        lam: Decay factor controlling hump location.

    Returns:
        Array of continuously-compounded zero rates.
    """
    tau = np.asarray(tau, dtype=np.float64)
    x = tau / lam
    exp_x = np.exp(-x)
    factor1 = np.where(x < 1e-10, 1.0, (1.0 - exp_x) / x)
    factor2 = factor1 - exp_x
    return beta0 + beta1 * factor1 + beta2 * factor2


def nelson_siegel_svensson(tau: np.ndarray, beta0: float, beta1: float,
                           beta2: float, beta3: float,
                           lam1: float, lam2: float) -> np.ndarray:
    """Nelson-Siegel-Svensson extended yield curve model.

    Adds a second hump component for more flexible curve fitting.

    Args:
        tau: Array of maturities.
        beta0: Level.
        beta1: Slope.
        beta2: First curvature.
        beta3: Second curvature.
        lam1: First decay factor.
        lam2: Second decay factor.

    Returns:
        Array of zero rates.
    """
    tau = np.asarray(tau, dtype=np.float64)
    x1 = tau / lam1
    x2 = tau / lam2
    exp1 = np.exp(-x1)
    exp2 = np.exp(-x2)
    f1 = np.where(x1 < 1e-10, 1.0, (1.0 - exp1) / x1)
    f2 = f1 - exp1
    f3 = np.where(x2 < 1e-10, 1.0, (1.0 - exp2) / x2) - exp2
    return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3


def discount_factor(tau: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Convert zero rates to discount factors.

    Args:
        tau: Maturities in years.
        rates: Continuously-compounded zero rates.

    Returns:
        Array of discount factors P(0, tau).
    """
    return np.exp(-np.asarray(rates) * np.asarray(tau))


def forward_rate(tau1: float, tau2: float, y1: float, y2: float) -> float:
    """Extract the instantaneous forward rate between two tenors.

    Args:
        tau1: Start maturity.
        tau2: End maturity.
        y1: Zero rate at tau1.
        y2: Zero rate at tau2.

    Returns:
        Forward rate f(tau1, tau2).
    """
    if abs(tau2 - tau1) < 1e-12:
        return y1
    return (y2 * tau2 - y1 * tau1) / (tau2 - tau1)


def fit_nelson_siegel(market_tenors: np.ndarray,
                      market_rates: np.ndarray) -> Tuple[float, float, float, float]:
    """Calibrate Nelson-Siegel parameters to market yield curve data.

    Args:
        market_tenors: Array of observed maturities.
        market_rates: Array of observed zero rates at those maturities.

    Returns:
        Tuple of (beta0, beta1, beta2, lambda).
    """
    market_tenors = np.asarray(market_tenors, dtype=np.float64)
    market_rates = np.asarray(market_rates, dtype=np.float64)

    def objective(params):
        b0, b1, b2, lam = params
        if lam <= 0.01:
            return 1e12
        fitted = nelson_siegel(market_tenors, b0, b1, b2, lam)
        return np.sum((fitted - market_rates) ** 2)

    r_long = market_rates[-1]
    r_short = market_rates[0]
    x0 = np.array([r_long, r_short - r_long, 0.0, 1.0])
    bounds = [(-0.1, 0.2), (-0.2, 0.2), (-0.2, 0.2), (0.01, 10.0)]

    result = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)
    return tuple(result.x)


def bootstrap_zeros(par_tenors: np.ndarray,
                    par_rates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap zero-coupon rates from par swap rates.

    Args:
        par_tenors: Array of swap tenors (must be annual: 1, 2, 3, ...).
        par_rates: Array of par swap rates.

    Returns:
        Tuple of (tenors, zero_rates).
    """
    par_tenors = np.asarray(par_tenors, dtype=np.float64)
    par_rates = np.asarray(par_rates, dtype=np.float64)
    n = par_tenors.shape[0]
    zeros = np.empty(n)
    dfs = np.empty(n)

    for i in range(n):
        c = par_rates[i]
        coupon_pv = sum(c * dfs[j] for j in range(i))
        dfs[i] = (1.0 - coupon_pv) / (1.0 + c)
        zeros[i] = -np.log(dfs[i]) / par_tenors[i]

    return par_tenors, zeros
