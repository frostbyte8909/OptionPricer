import numpy as np
from scipy.optimize import brentq
from optionpricer.models.binomial import build_tree
from optionpricer.analytics.greeks import greeks
from optionpricer.core import OptionContract, MarketState

try:
    from py_lets_be_rational import implied_volatility_from_a_transformed_rational_guess
    HAS_LBR = True
except ImportError:
    HAS_LBR = False

def implied_vol(market_price: float, contract: OptionContract, market: MarketState, N: int = 100) -> float:
    S, K, T, r, q = float(market.spot), float(contract.strike), float(contract.expiry), float(market.rate), float(market.dividend)
    
    if HAS_LBR and not contract.american:
        F = S * np.exp((r - q) * T)
        D = np.exp(-r * T)
        c = 1 if contract.option_type == "call" else -1
        iv = implied_volatility_from_a_transformed_rational_guess(market_price / D, F, K, T, c)
        if iv > 0:
            return iv

    is_near_atm = 0.8 < (S / K) < 1.2
    if is_near_atm:
        sigma_guess = 0.2
        tol = 1e-6
        for _ in range(100):
            m_state = MarketState(S, r, sigma_guess, q)
            price = build_tree(contract, m_state, N)
            diff = price - market_price
            if abs(diff) < tol:
                return sigma_guess

            vega = greeks(contract, m_state, N).get('vega', 0.0)
            if abs(vega) < 1e-8:
                break
                
            sigma_guess -= diff / (vega * 100)
            if sigma_guess <= 0:
                break

    def objective(sigma):
        return build_tree(contract, MarketState(S, r, sigma, q), N) - market_price

    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan
