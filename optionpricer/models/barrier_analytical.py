import numpy as np
from scipy.stats import norm
from typing import Union
from optionpricer.core import OptionContract, MarketState, ExoticType, OptionType

def _phi(x):
    return norm.cdf(x)

def _barrier_analytical_single(S, K, H, T, r, q, sigma, option_type, exotic_type, rebate):
    b = r - q
    
    if (exotic_type == ExoticType.BARRIER_DO and S <= H) or \
       (exotic_type == ExoticType.BARRIER_DI and S <= H) or \
       (exotic_type == ExoticType.BARRIER_UO and S >= H) or \
       (exotic_type == ExoticType.BARRIER_UI and S >= H):
        return rebate
    
    mu = (b - (sigma**2) / 2) / (sigma**2)
    lambda_ = np.sqrt(mu**2 + 2 * r / (sigma**2))
    Z = np.log(H / S) / (sigma * np.sqrt(T)) + lambda_ * sigma * np.sqrt(T)
    
    x1 = np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    x2 = np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    
    y1 = np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    y2 = np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)
    
    phi = 1 if option_type == OptionType.CALL else -1
    eta = 1 if exotic_type in (ExoticType.BARRIER_DO, ExoticType.BARRIER_DI) else -1
    
    def A(phi):
        return phi * S * np.exp((b - r) * T) * _phi(phi * x1) - phi * K * np.exp(-r * T) * _phi(phi * x1 - phi * sigma * np.sqrt(T))
        
    def B(phi):
        return phi * S * np.exp((b - r) * T) * _phi(phi * x2) - phi * K * np.exp(-r * T) * _phi(phi * x2 - phi * sigma * np.sqrt(T))
        
    def C(phi, eta):
        return phi * S * np.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _phi(eta * y1) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu) * _phi(eta * y1 - eta * sigma * np.sqrt(T))
        
    def D(phi, eta):
        return phi * S * np.exp((b - r) * T) * (H / S)**(2 * (mu + 1)) * _phi(eta * y2) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu) * _phi(eta * y2 - eta * sigma * np.sqrt(T))

    def E(eta):
        return rebate * np.exp(-r * T) * (_phi(eta * x2 - eta * sigma * np.sqrt(T)) - (H / S)**(2 * mu) * _phi(eta * y2 - eta * sigma * np.sqrt(T)))
        
    def F(eta):
        return rebate * ((H / S)**(mu + lambda_) * _phi(eta * Z) + (H / S)**(mu - lambda_) * _phi(eta * Z - 2 * eta * lambda_ * sigma * np.sqrt(T)))

    price = 0.0
    
    if exotic_type == ExoticType.BARRIER_DI:
        if K > H:
            if option_type == OptionType.CALL:
                price = C(1, 1) + E(1)
            else:
                price = B(-1) - C(-1, 1) + D(-1, 1) + E(1)
        else:
            if option_type == OptionType.CALL:
                price = A(1) - B(1) + D(1, 1) + E(1)
            else:
                price = A(-1) + E(1)
                
    elif exotic_type == ExoticType.BARRIER_DO:
        if K > H:
            if option_type == OptionType.CALL:
                price = A(1) - C(1, 1) + F(1)
            else:
                price = A(-1) - B(-1) + C(-1, 1) - D(-1, 1) + F(1)
        else:
            if option_type == OptionType.CALL:
                price = B(1) - D(1, 1) + F(1)
            else:
                price = F(1)

    elif exotic_type == ExoticType.BARRIER_UI:
        if K > H:
            if option_type == OptionType.CALL:
                price = A(1) + E(-1)
            else:
                price = A(-1) - B(-1) + D(-1, -1) + E(-1)
        else:
            if option_type == OptionType.CALL:
                price = B(1) - C(1, -1) + D(1, -1) + E(-1)
            else:
                price = C(-1, -1) + E(-1)
                
    elif exotic_type == ExoticType.BARRIER_UO:
        if K > H:
            if option_type == OptionType.CALL:
                price = F(-1)
            else:
                price = B(-1) - D(-1, -1) + F(-1)
        else:
            if option_type == OptionType.CALL:
                price = A(1) - B(1) + C(1, -1) - D(1, -1) + F(-1)
            else:
                price = A(-1) - C(-1, -1) + F(-1)
                
    return price

def barrier_analytical(contract: OptionContract, market: MarketState) -> Union[float, np.ndarray]:
    br = np.broadcast(market.spot, contract.strike, contract.expiry, market.rate, market.volatility, market.dividend, contract.barrier_level, contract.rebate)
    S = np.broadcast_to(market.spot, br.shape).astype(float)
    K = np.broadcast_to(contract.strike, br.shape).astype(float)
    T = np.broadcast_to(contract.expiry, br.shape).astype(float)
    r = np.broadcast_to(market.rate, br.shape).astype(float)
    sig = np.broadcast_to(market.volatility, br.shape).astype(float)
    q = np.broadcast_to(market.dividend, br.shape).astype(float)
    H = np.broadcast_to(contract.barrier_level, br.shape).astype(float)
    rebate = np.broadcast_to(contract.rebate, br.shape).astype(float)
    
    result = np.empty_like(S)
    it = np.nditer([S, K, H, T, r, q, sig, rebate, result], op_flags=[['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['readonly'], ['writeonly']])
    
    for s_i, k_i, h_i, t_i, r_i, q_i, sig_i, reb_i, res_i in it:
        res_i[...] = _barrier_analytical_single(s_i, k_i, h_i, t_i, r_i, q_i, sig_i, contract.option_type, contract.exotic_type, reb_i)

    if result.ndim == 0:
        return float(result.item())
    return result
