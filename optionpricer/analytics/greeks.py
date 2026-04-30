import numpy as np
from optionpricer.models.binomial import build_tree
from optionpricer.core import OptionContract, MarketState
import copy

def greeks(contract: OptionContract, market: MarketState, N: int = 100) -> dict[str, float]:
    S, K, T, r, sigma, q = float(market.spot), float(contract.strike), float(contract.expiry), float(market.rate), float(market.volatility), float(market.dividend)
    
    if not contract.american:
        from scipy.stats import norm
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1) if contract.option_type == "call" else norm.cdf(d1) - 1.0
        
        delta = np.exp(-q * T) * cdf_d1
        gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T) / 100.0
        
        if contract.option_type == "call":
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
        else:
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0
            
        return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega), "theta": float(theta), "rho": float(rho)}

    bump_sigma = 1e-4
    bump_r     = 1e-4
    bump_T     = 1.0 / 365.0

    f = build_tree(contract, market, N)
    
    ds = S * 0.01
    f_up = build_tree(contract, MarketState(S + ds, r, sigma, q), N)
    f_dn = build_tree(contract, MarketState(S - ds, r, sigma, q), N)
    delta = (f_up - f_dn) / (2 * ds)
    gamma = (f_up - 2 * f + f_dn) / (ds ** 2)

    c_theta = copy.deepcopy(contract)
    c_theta.expiry = T - bump_T
    theta = (build_tree(c_theta, market, N) - f) / bump_T / 365.0

    vega = (build_tree(contract, MarketState(S, r, sigma + bump_sigma, q), N) - f) / bump_sigma / 100.0

    rho = (build_tree(contract, MarketState(S, r + bump_r, sigma, q), N) - f) / bump_r / 100.0

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
