import numpy as np
from scipy.optimize import brentq
from optionpricer.models.binomial import build_tree
from optionpricer.analytics.greeks import greeks

try:
    from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
    HAS_LBR = True
except ImportError:
    HAS_LBR = False

def implied_vol(market_price: float, S: float, K: float, T: float, r: float, N: int = 100, option_type: str = "call", american: bool = False) -> float:
    """
    Calculate the implied volatility of an option given its market price.
    Uses 'Let's Be Rational' if available for European options, otherwise falls back to a Newton-Raphson 
    or Brent's method solver using the binomial tree.

    Args:
        market_price (float): The observed market price of the option.
        S (float): Current asset price.
        K (float): Strike price of the option.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate (annualized).
        N (int, optional): Number of steps in the binomial tree (if used). Defaults to 100.
        option_type (str, optional): 'call' for Call option, 'put' for Put option. Defaults to 'call'.
        american (bool, optional): If True, calculates implied vol for an American option. Defaults to False.

    Returns:
        float: The implied volatility (sigma).
    """
    if HAS_LBR and not american:
        F    = S * np.exp(r * T)
        D    = np.exp(-r * T)
        q    = 1 if option_type == "call" else -1
        iv   = implied_volatility_from_a_transformed_rational_guess(
                   market_price / D, F, K, T, q)
        if iv > 0:
            return iv

    is_near_atm = 0.8 < (S / K) < 1.2
    
    if is_near_atm and not american:
        sigma = 0.2
        for _ in range(20):
            price = build_tree(S, K, T, r, sigma, N, option_type, american)
            v     = greeks(S, K, T, r, sigma, N, option_type, american)["vega"]
            if abs(v) < 1e-6:
                break
            sigma -= (price - market_price) / v
            if abs(price - market_price) < 1e-8:
                return sigma

    # BrentQ for edge-cases and fallback
    objective = lambda s: build_tree(S, K, T, r, s, N, option_type, american) - market_price
    return brentq(objective, 1e-4, 10.0)
