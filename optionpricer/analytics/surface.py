import numpy as np
from scipy.interpolate import RectBivariateSpline
from math import erfc, sqrt, exp, log


_SQRT2 = sqrt(2.0)
_INV_SQRT2PI = 1.0 / sqrt(2.0 * np.pi)


def _phi(x: float) -> float:
    return 0.5 * erfc(-x / _SQRT2)


def _npdf(x: float) -> float:
    return _INV_SQRT2PI * exp(-0.5 * x * x)


def dupire_local_vol(strikes: np.ndarray, expiries: np.ndarray,
                     iv_surface: np.ndarray, S: float, r: float,
                     q: float) -> np.ndarray:
    """Compute the Dupire local volatility surface from an implied vol grid.

    Uses cubic spline interpolation of call prices to compute the partial
    derivatives dC/dT, dC/dK, and d2C/dK2 required by Dupire's formula.

    Args:
        strikes: 1D array of strikes (sorted ascending).
        expiries: 1D array of expiry tenors (sorted ascending).
        iv_surface: 2D array of implied vols, shape (len(expiries), len(strikes)).
        S: Current spot price.
        r: Risk-free rate.
        q: Continuous dividend yield.

    Returns:
        2D array of local volatilities, same shape as iv_surface.
    """
    nT, nK = iv_surface.shape
    call_surface = np.empty_like(iv_surface)

    for i in range(nT):
        T_i = expiries[i]
        sqT = sqrt(T_i)
        df_r = exp(-r * T_i)
        df_q = exp(-q * T_i)
        for j in range(nK):
            K_j = strikes[j]
            sig = iv_surface[i, j]
            d1 = (log(S / K_j) + (r - q + 0.5 * sig * sig) * T_i) / (sig * sqT)
            d2 = d1 - sig * sqT
            call_surface[i, j] = S * df_q * _phi(d1) - K_j * df_r * _phi(d2)

    kx = min(3, nT - 1)
    ky = min(3, nK - 1)
    spline = RectBivariateSpline(expiries, strikes, call_surface, kx=kx, ky=ky)

    dC_dT = spline(expiries, strikes, dx=1, dy=0)
    dC_dK = spline(expiries, strikes, dx=0, dy=1)
    d2C_dK2 = spline(expiries, strikes, dx=0, dy=2)

    local_var = np.empty_like(iv_surface)
    for i in range(nT):
        for j in range(nK):
            K_j = strikes[j]
            numer = dC_dT[i, j] + (r - q) * K_j * dC_dK[i, j] + q * call_surface[i, j]
            denom = 0.5 * K_j * K_j * d2C_dK2[i, j]
            if denom < 1e-14:
                local_var[i, j] = iv_surface[i, j] ** 2
            else:
                local_var[i, j] = max(numer / denom, 1e-8)

    return np.sqrt(local_var)


def arbitrage_check(strikes: np.ndarray, expiries: np.ndarray,
                    iv_surface: np.ndarray, S: float, r: float,
                    q: float) -> dict:
    """Validate arbitrage-free conditions on a vol surface.

    Checks calendar spread (dC/dT >= 0) and butterfly (d2C/dK2 >= 0)
    constraints across the surface grid.

    Args:
        strikes: 1D array of strikes.
        expiries: 1D array of expiries.
        iv_surface: 2D implied vol grid, shape (len(expiries), len(strikes)).
        S: Spot price.
        r: Risk-free rate.
        q: Dividend yield.

    Returns:
        Dict with keys 'calendar_violations' and 'butterfly_violations',
        each containing a list of (T_idx, K_idx) tuples where violations occur.
    """
    nT, nK = iv_surface.shape
    call_surface = np.empty_like(iv_surface)

    for i in range(nT):
        T_i = expiries[i]
        sqT = sqrt(T_i)
        df_r = exp(-r * T_i)
        df_q = exp(-q * T_i)
        for j in range(nK):
            K_j = strikes[j]
            sig = iv_surface[i, j]
            d1 = (log(S / K_j) + (r - q + 0.5 * sig * sig) * T_i) / (sig * sqT)
            d2 = d1 - sig * sqT
            call_surface[i, j] = S * df_q * _phi(d1) - K_j * df_r * _phi(d2)

    calendar = []
    for i in range(nT - 1):
        for j in range(nK):
            if call_surface[i + 1, j] - call_surface[i, j] < -1e-10:
                calendar.append((i, j))

    butterfly = []
    for i in range(nT):
        for j in range(1, nK - 1):
            d2C = (call_surface[i, j + 1] - 2 * call_surface[i, j] + call_surface[i, j - 1])
            if d2C < -1e-10:
                butterfly.append((i, j))

    return {"calendar_violations": calendar, "butterfly_violations": butterfly}
