import numpy as np
from scipy.optimize import brentq
from optionpricer.models.binomial import build_tree
from optionpricer.analytics.greeks import greeks

try:
    from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
    HAS_LBR = True
except ImportError:
    HAS_LBR = False

def implied_vol(market_price: float, S: float, K: float, T: float, r: float, q: float = 0.0, N: int = 100, option_type: str = "call", american: bool = False) -> float:
    """
    Calculate the implied volatility of an option given its market price.
    Uses 'Let's Be Rational' if available for European options, otherwise falls back to a Newton-Raphson 
    or Brent's method solver using the binomial tree. Supports continuous dividends.

    Args:
        market_price (float): The observed market price of the option.
        S (float): Current asset price.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate (annualized).
        q (float, optional): Continuous dividend yield. Defaults to 0.0.
        N (int, optional): Number of steps in the binomial tree (if used). Defaults to 100.
        option_type (str, optional): 'call' for Call option, 'put' for Put option. Defaults to 'call'.
        american (bool, optional): If True, calculates implied vol for an American option. Defaults to False.

    Returns:
        float: The implied volatility (sigma).
    """
    if HAS_LBR and not american:
        F    = S * np.exp((r - q) * T)
        D    = np.exp(-r * T)
        c    = 1 if option_type == "call" else -1
        iv   = implied_volatility_from_a_transformed_rational_guess(
                   market_price / D, F, K, T, c)
        if iv > 0:
            return iv

    is_near_atm = 0.8 < (S / K) < 1.2
    
    if is_near_atm:
        sigma_guess = 0.2
        max_iter = 100
        tol = 1e-6

        for _ in range(max_iter):
            price = build_tree(S, K, T, r, sigma_guess, q, N, option_type, american)
            diff = price - market_price

            if abs(diff) < tol:
                return sigma_guess

            vega = greeks(S, K, T, r, sigma_guess, q, N, option_type, american).get('vega', 0.0)
            if vega == 0.0:
                break

            sigma_guess -= diff / (vega * 100)
            if sigma_guess <= 0:
                sigma_guess = 1e-4

    def objective(sigma):
        return build_tree(S, K, T, r, sigma, q, N, option_type, american) - market_price

    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan
