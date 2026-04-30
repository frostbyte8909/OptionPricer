import numpy as np
from optionpricer.models.binomial import build_tree

def greeks(S, K, T, r, sigma, q=0.0, N=100, option_type="call", american=False):
    if not american:
        from scipy.stats import norm
        S = np.asarray(S)
        K = np.asarray(K)
        T = np.asarray(T)
        r = np.asarray(r)
        sigma = np.asarray(sigma)
        q = np.asarray(q)
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0
        
        delta = np.exp(-q * T) * cdf_d1
        gamma = np.exp(-q * T) * pdf_d1 / (S * sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * pdf_d1 * np.sqrt(T)
        
        if option_type == "call":
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) + q * S * np.exp(-q * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            theta = (- (S * sigma * np.exp(-q * T) * pdf_d1) / (2 * np.sqrt(T)) - q * S * np.exp(-q * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
        return {"delta": float(delta) if delta.ndim == 0 else delta, "gamma": float(gamma) if gamma.ndim == 0 else gamma, "vega": float(vega) if vega.ndim == 0 else vega, "theta": float(theta) if theta.ndim == 0 else theta, "rho": float(rho) if rho.ndim == 0 else rho}

    bump_sigma = 0.01
    bump_r     = 0.001
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1 / u

    Su = S * u
    Sd = S * d

    fu  = build_tree(Su, K, T - dt,   r, sigma, q, N, option_type, american)
    fd  = build_tree(Sd, K, T - dt,   r, sigma, q, N, option_type, american)
    f   = build_tree(S,  K, T - dt,   r, sigma, q, N, option_type, american)
    fud = build_tree(S,  K, T - 2*dt, r, sigma, q, N, option_type, american)

    delta = (fu - fd) / (Su - Sd)
    gamma = ((fu - f) / (Su - S) - (f - fd) / (S - Sd)) / (0.5 * (Su - Sd))
    theta = (fud - f) / (2 * dt) / 365

    vega = (build_tree(S, K, T, r, sigma + bump_sigma, q, N, option_type, american)
          - build_tree(S, K, T, r, sigma - bump_sigma, q, N, option_type, american)) / (2 * bump_sigma)

    price_up   = build_tree(S, K, T, r + bump_r, sigma, q, N, option_type, american)
    price_down = build_tree(S, K, T, r - bump_r, sigma, q, N, option_type, american)
    rho = (price_up - price_down) / (2 * bump_r) / 100

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
