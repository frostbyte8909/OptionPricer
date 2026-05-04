import numpy as np
from scipy.stats import norm
from typing import Union
from optionpricer.core import OptionContract, MarketState

def _phi(S, gamma, H, I, r, b, sigma, T):
    lambda_ = (-b / sigma**2 + 0.5) + np.sqrt((-b / sigma**2 + 0.5)**2 + 2 * r / sigma**2)
    kappa = 2 * b / sigma**2 + (2 * lambda_ - 1)
    
    d = -(np.log(S / H) + (b + (gamma - 0.5) * sigma**2) * T) / (sigma * np.sqrt(T))
    
    val = np.exp(lambda_ * T) * (S**gamma) * (norm.cdf(d) - (I / S)**kappa * norm.cdf(d - 2 * np.log(I / S) / (sigma * np.sqrt(T))))
    return val

def _bjerksund_stensland_call(S, K, T, r, b, sigma):
    if b >= r:
        d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    beta = (0.5 - b / sigma**2) + np.sqrt((b / sigma**2 - 0.5)**2 + 2 * r / sigma**2)
    B_infinity = beta / (beta - 1) * K
    B0 = max(K, r / (r - b) * K)
    
    ht = -(b + 0.5 * sigma**2) * T / (sigma * np.sqrt(T))
    I = B0 + (B_infinity - B0) * (1 - np.exp(ht))
    
    alpha = (I - K) * I**(-beta)
    
    if S >= I:
        return S - K
    else:
        d1 = (np.log(S / I) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        eur_price = S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        
        return eur_price + alpha * S**beta * norm.cdf(-d1)

def bjerksund_stensland_american(contract: OptionContract, market: MarketState) -> Union[float, np.ndarray]:
    br = np.broadcast(market.spot, contract.strike, contract.expiry, market.rate, market.volatility, market.dividend)
    S = np.broadcast_to(market.spot, br.shape).astype(float)
    K = np.broadcast_to(contract.strike, br.shape).astype(float)
    T = np.broadcast_to(contract.expiry, br.shape).astype(float)
    r = np.broadcast_to(market.rate, br.shape).astype(float)
    sig = np.broadcast_to(market.volatility, br.shape).astype(float)
    q = np.broadcast_to(market.dividend, br.shape).astype(float)
    
    b = r - q
    
    result = np.empty_like(S)
    it = np.nditer([S, K, T, r, b, sig, result], op_flags=[['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['writeonly']])
    
    if contract.option_type.value == "call":
        for s_i, k_i, t_i, r_i, b_i, sig_i, res_i in it:
            res_i[...] = _bjerksund_stensland_call(s_i, k_i, t_i, r_i, b_i, sig_i)
    else:
        for s_i, k_i, t_i, r_i, b_i, sig_i, res_i in it:
            res_i[...] = _bjerksund_stensland_call(k_i, s_i, t_i, r_i - b_i, -b_i, sig_i)

    if result.ndim == 0:
        return float(result.item())
    return result
