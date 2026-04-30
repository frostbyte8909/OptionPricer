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
    if HAS_LBR and not american:
        F    = S * np.exp((r - q) * T)
        D    = np.exp(-r * T)
        c    = 1 if option_type == "call" else -1
        iv   = implied_volatility_from_a_transformed_rational_guess(market_price / D, F, K, T, c)
        if iv > 0:
            return iv

    is_near_atm = 0.8 < (S / K) < 1.2
    
    if is_near_atm:
        sigma_guess = 0.2
        tol = 1e-6

        for _ in range(100):
            price = build_tree(S, K, T, r, sigma_guess, q, N, option_type, american)
            diff = price - market_price

            if abs(diff) < tol:
                return sigma_guess

            vega = greeks(S, K, T, r, sigma_guess, q, N, option_type, american).get('vega', 0.0)
            if abs(vega) < 1e-8:
                break

            sigma_guess -= diff / (vega * 100)
            if sigma_guess <= 0:
                break

    def objective(sigma):
        return build_tree(S, K, T, r, sigma, q, N, option_type, american) - market_price

    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan
