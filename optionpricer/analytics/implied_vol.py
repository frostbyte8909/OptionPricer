import numpy as np
from scipy.optimize import brentq
from optionpricer.models.binomial import build_tree
from optionpricer.analytics.greeks import greeks

try:
    from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
    HAS_LBR = True
except ImportError:
    HAS_LBR = False

def implied_vol(market_price, S, K, T, r, N=100, option_type="call", american=False):
    if HAS_LBR and not american:
        F    = S * np.exp(r * T)
        D    = np.exp(-r * T)
        q    = 1 if option_type == "call" else -1
        iv   = implied_volatility_from_a_transformed_rational_guess(
                   market_price / D, F, K, T, q)
        if iv > 0:
            return iv

    sigma = 0.2
    for _ in range(100):
        price = build_tree(S, K, T, r, sigma, N, option_type, american)
        v     = greeks(S, K, T, r, sigma, N, option_type, american)["vega"]
        if abs(v) < 1e-6:
            break
        sigma -= (price - market_price) / v
        if abs(price - market_price) < 1e-8:
            return sigma

    objective = lambda s: build_tree(S, K, T, r, s, N, option_type, american) - market_price
    return brentq(objective, 1e-4, 10.0)
